"""Módulo de audio: generación de tonos sin chasquidos y captura por búfer circular.

Diseñado para alta estabilidad en Linux (ALSA/PulseAudio/PipeWire):
- Búfer circular sin bloqueos y sin asignación de memoria en el callback.
- Rampas suaves de ganancia (anti-clic) y limitador suave contra saturación.
- Detección y recomendación de dispositivos, con soporte de modo seguro (offline).
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
except Exception as _exc:
    sd = None

logger = logging.getLogger("chladni_lab.audio")

SAMPLE_RATE = 44100
DEFAULT_BLOCK_SIZE = 2048


class LockFreeRingBuffer:
    """Búfer circular continuo para recepción de muestras de audio.

    El hilo de audio escribe de forma secuencial sin llamadas a malloc ni np.roll.
    El hilo de la interfaz toma una instantánea (snapshot) de las últimas N muestras.
    """

    def __init__(self, capacity: int):
        self.capacity = max(capacity, 1024)
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx = 0

    def write(self, data: np.ndarray) -> None:
        """Escribe un bloque de datos en el búfer circular sin reasignar memoria."""
        n = len(data)
        if n == 0:
            return

        if n >= self.capacity:
            # Los datos entrantes superan la capacidad total: conservar solo los últimos
            self.buffer[:] = data[-self.capacity:]
            self.write_idx = 0
            return

        w = self.write_idx
        end_idx = w + n
        if end_idx <= self.capacity:
            self.buffer[w:end_idx] = data
        else:
            first_part = self.capacity - w
            self.buffer[w:self.capacity] = data[:first_part]
            self.buffer[: n - first_part] = data[first_part:]

        self.write_idx = (self.write_idx + n) % self.capacity

    def snapshot(self, n: Optional[int] = None) -> np.ndarray:
        """Obtiene una copia consistente de las últimas n muestras."""
        if n is None or n > self.capacity:
            n = self.capacity
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        w = self.write_idx
        out = np.empty(n, dtype=np.float32)
        if w >= n:
            out[:] = self.buffer[w - n : w]
        else:
            first_part = n - w
            out[:first_part] = self.buffer[self.capacity - first_part : self.capacity]
            out[first_part:] = self.buffer[:w]
        return out


def list_audio_devices(kind: str = "output") -> List[Dict[str, Any]]:
    """Lista dispositivos disponibles de forma amigable para Windows y Linux.

    kind: 'input' o 'output'.
    Devuelve lista de diccionarios con info legible y si es recomendado.
    """
    if sd is None:
        return []

    devices = []
    try:
        all_devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_pair = sd.default.device
        default_idx = default_pair[0 if kind == "input" else 1] if default_pair is not None else None
    except Exception as exc:
        logger.warning(f"No se pudieron consultar los dispositivos de audio: {exc}")
        return []

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"

    for idx, d in enumerate(all_devs):
        chans = d.get(channel_key, 0)
        if chans <= 0:
            continue

        name = d.get("name", f"Dispositivo {idx}")
        sr = int(d.get("default_samplerate", SAMPLE_RATE))
        hostapi_idx = d.get("hostapi", 0)
        hostapi_name = ""
        if isinstance(hostapis, (list, tuple)) and 0 <= hostapi_idx < len(hostapis):
            hostapi_name = hostapis[hostapi_idx].get("name", "")

        is_default = (idx == default_idx)
        name_low = name.lower()
        # En Linux: pulse/pipewire/default; en Windows: Asignador de sonido Microsoft / primario
        is_preferred_virtual = (
            ("pulse" in name_low)
            or ("pipewire" in name_low)
            or ("default" in name_low and "sysdefault" not in name_low)
            or ("asignador de sonido" in name_low)
            or ("sound mapper" in name_low)
        )
        recommended = is_default or is_preferred_virtual

        label_parts = []
        if recommended:
            label_parts.append("[Recomendado]")
        label_parts.append(f"{idx}: {name}")
        if hostapi_name and len(hostapis) > 1:
            label_parts.append(f"[{hostapi_name}]")
        label_parts.append(f"({chans} ch, {sr} Hz)")

        devices.append({
            "index": idx,
            "name": name,
            "hostapi": hostapi_name,
            "channels": chans,
            "samplerate": sr,
            "is_default": is_default,
            "recommended": recommended,
            "label": " ".join(label_parts),
        })

    # Ordenar poniendo primero el predeterminado del sistema, luego recomendados y luego por índice
    devices.sort(key=lambda x: (not x["is_default"], not x["recommended"], x["index"]))
    return devices


class ToneGenerator:
    """Generador senoidal continuo con barrido por muestra, rampa suave anti-clic y limitador suave."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self._frequency = 200.0
        self._volume = 0.15  # Inicio con volumen seguro
        self._playing = False
        self._phase = 0.0
        self._cur_f = self._frequency
        self._cur_a = 0.0  # Ganancia actual aplicada

        # Estado del barrido interno (generado muestra a muestra en el hilo de audio)
        self._sweep_active = False
        self._sweep_paused = False
        self._sweep_f0 = 100.0
        self._sweep_f1 = 1000.0
        self._sweep_duration = 45.0
        self._sweep_log = True
        self._sweep_pingpong = False
        self._sweep_samples_elapsed = 0
        self._sweep_direction = 1  # 1: ida, -1: vuelta

        self._stream: Optional[Any] = None
        self._device: Optional[int] = None
        self._scratch = np.zeros(DEFAULT_BLOCK_SIZE * 4, dtype=np.float32)

    @property
    def frequency(self) -> float:
        return self._frequency

    @frequency.setter
    def frequency(self, val: float):
        self._frequency = max(10.0, min(20000.0, float(val)))

    @property
    def current_frequency(self) -> float:
        """Devuelve la frecuencia acústica instantánea actual del generador."""
        return float(self._cur_f)

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, val: float):
        self._volume = max(0.0, min(1.0, float(val)))

    @property
    def playing(self) -> bool:
        return self._playing

    @playing.setter
    def playing(self, val: bool):
        self._playing = bool(val)

    @property
    def sweep_active(self) -> bool:
        return self._sweep_active

    @property
    def sweep_progress(self) -> float:
        total = max(1, int(self._sweep_duration * self.sr))
        return min(1.0, max(0.0, self._sweep_samples_elapsed / total))

    def start_sweep(
        self,
        f0: float,
        f1: float,
        duration: float,
        log: bool = True,
        pingpong: bool = False
    ) -> None:
        """Inicia un barrido continuo generado muestra a muestra sin chasquidos."""
        self._sweep_f0 = float(f0)
        self._sweep_f1 = float(f1)
        self._sweep_duration = max(2.0, float(duration))
        self._sweep_log = bool(log)
        self._sweep_pingpong = bool(pingpong)
        self._sweep_samples_elapsed = 0
        self._sweep_direction = 1
        self._cur_f = float(f0)
        self._sweep_paused = False
        self._sweep_active = True
        self._playing = True

    def pause_sweep(self) -> None:
        self._sweep_paused = True

    def resume_sweep(self) -> None:
        self._sweep_paused = False

    def stop_sweep(self) -> None:
        self._sweep_active = False
        self._sweep_paused = False

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if len(self._scratch) < frames:
            self._scratch = np.zeros(frames * 2, dtype=np.float32)

        sr = self.sr
        cur_f = self._cur_f
        cur_a = self._cur_a
        target_a = self._volume if self._playing else 0.0

        if self._sweep_active and not self._sweep_paused:
            total_samples = max(1, int(self._sweep_duration * sr))
            elapsed = self._sweep_samples_elapsed

            # Generar el vector de tiempo continuo muestra a muestra para este bloque
            sample_indices = elapsed + np.arange(frames, dtype=np.float64)
            progress = np.clip(sample_indices / total_samples, 0.0, 1.0)

            if self._sweep_pingpong and self._sweep_direction == -1:
                progress = 1.0 - progress

            f0 = self._sweep_f0
            f1 = self._sweep_f1

            if self._sweep_log and f0 > 0 and f1 > 0:
                f_arr = f0 * ((f1 / f0) ** progress)
            else:
                f_arr = f0 + (f1 - f0) * progress

            self._sweep_samples_elapsed += frames
            if self._sweep_samples_elapsed >= total_samples:
                if self._sweep_pingpong and self._sweep_direction == 1:
                    # Invertir para la vuelta
                    self._sweep_direction = -1
                    self._sweep_samples_elapsed = 0
                else:
                    self._sweep_active = False

            self._cur_f = float(f_arr[-1])
            self._frequency = self._cur_f
        else:
            # Modo estático o manual: rampa de frecuencia suave entre bloques
            target_f = self._frequency
            df = (target_f - cur_f) / frames
            f_arr = cur_f + np.arange(frames, dtype=np.float64) * df
            self._cur_f = target_f

        # Acumulación continua de fase
        dphi = (2.0 * np.pi / sr) * f_arr
        phases = self._phase + np.cumsum(dphi)
        self._phase = float(phases[-1] % (2.0 * np.pi))

        # Rampa suave de amplitud continua para eliminar transitorios y clics
        da = (target_a - cur_a) / frames
        gains = cur_a + np.arange(frames, dtype=np.float32) * da
        self._cur_a = target_a

        # Generar onda senoidal en búfer de trabajo preasignado
        np.sin(phases, out=self._scratch[:frames])
        self._scratch[:frames] *= gains

        # Limitador suave no lineal (soft saturation tanh)
        np.tanh(self._scratch[:frames], out=self._scratch[:frames])

        # Asignar a los canales de salida sin reasignar memoria
        outdata[:, 0] = self._scratch[:frames]
        if outdata.shape[1] > 1:
            outdata[:, 1] = self._scratch[:frames]

    def start(self, device: Optional[int] = None) -> Tuple[bool, str]:
        """Inicia el stream de salida con latencia configurada para prevenir subdesbordes."""
        if sd is None:
            return False, "Módulo sounddevice no disponible."

        self.stop()
        self._device = device

        try:
            try:
                dinfo = sd.query_devices(device, "output") if device is None else sd.query_devices(device)
                self.sr = int(dinfo.get("default_samplerate", self.sr))
            except Exception:
                pass

            self._stream = sd.OutputStream(
                samplerate=self.sr,
                channels=2,
                dtype="float32",
                device=device,
                callback=self._callback,
                blocksize=DEFAULT_BLOCK_SIZE,
                latency="high",
            )
            self._stream.start()
            logger.info(f"ToneGenerator iniciado en dispositivo {device} a {self.sr} Hz (blocksize={DEFAULT_BLOCK_SIZE})")
            return True, f"Audio iniciado ({self.sr} Hz)"
        except Exception as exc:
            logger.warning(f"Fallo al abrir OutputStream en dispositivo {device}: {exc}")
            self._stream = None
            return False, str(exc)

    def stop(self) -> None:
        """Detiene el stream de forma segura."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug(f"Excepción controlada al cerrar stream: {exc}")
            finally:
                self._stream = None
                self._cur_a = 0.0


class InputMonitor:
    """Monitor de entrada por cable 3.5 mm con búfer circular sin bloqueos."""

    def __init__(self, sr: int = SAMPLE_RATE, seconds: float = 0.5):
        self.sr = sr
        self._seconds = seconds
        self.ring = LockFreeRingBuffer(capacity=int(sr * seconds))
        self._stream: Optional[Any] = None
        self._device: Optional[int] = None
        self._channels: int = 1

    @property
    def buf(self) -> np.ndarray:
        """Propiedad de compatibilidad hacia atrás con el código previo."""
        return self.ring.snapshot()

    def snapshot(self, n: Optional[int] = None) -> np.ndarray:
        return self.ring.snapshot(n)

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # Extraer primer canal (mono o canal izquierdo de jack de línea)
        channel_data = indata[:, 0]
        self.ring.write(channel_data)

    def start(self, device: Optional[int] = None) -> Tuple[bool, str]:
        """Inicia la captura intentando mono y recurriendo a estéreo si el dispositivo lo requiere."""
        if sd is None:
            return False, "Módulo sounddevice no disponible."

        self.stop()
        self._device = device

        try:
            dinfo = sd.query_devices(device, "input") if device is None else sd.query_devices(device)
            new_sr = int(dinfo.get("default_samplerate", self.sr))
            if new_sr != self.sr:
                self.sr = new_sr
                self.ring = LockFreeRingBuffer(capacity=int(self.sr * self._seconds))
        except Exception:
            pass

        # Intentar primero con 1 canal, y si falla probar con 2 canales (común en tarjetas ALSA)
        for ch in (1, 2):
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sr,
                    channels=ch,
                    dtype="float32",
                    device=device,
                    callback=self._callback,
                    blocksize=DEFAULT_BLOCK_SIZE,
                    latency="high",
                )
                self._stream.start()
                self._channels = ch
                logger.info(f"InputMonitor iniciado en dispositivo {device} ({ch} canales, {self.sr} Hz)")
                return True, f"Captura iniciada ({ch} ch, {self.sr} Hz)"
            except Exception as exc:
                logger.debug(f"Intento con {ch} canales falló: {exc}")
                if ch == 2:
                    self._stream = None
                    return False, str(exc)

        return False, "No se pudo inicializar la entrada de audio."

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug(f"Excepción controlada al cerrar InputStream: {exc}")
            finally:
                self._stream = None

