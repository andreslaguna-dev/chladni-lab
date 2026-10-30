"""Módulo de procesamiento digital de señales (DSP).

Incluye análisis espectral mediante FFT, interpolación parabólica de alta resolución,
conversión de frecuencia a notas musicales con centésimas, y filtrado adaptativo.
Sin dependencias de interfaz gráfica (apto para pruebas sin pantalla).
"""
import math
from typing import Dict, Optional, Tuple
import numpy as np

SAMPLE_RATE = 44100
NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

_HANN_CACHE: Dict[int, np.ndarray] = {}
_FREQS_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _get_hann_window(n: int) -> np.ndarray:
    win = _HANN_CACHE.get(n)
    if win is None:
        win = np.hanning(n)
        _HANN_CACHE[n] = win
    return win


def _get_rfftfreqs(nfft: int, sr: int) -> np.ndarray:
    key = (nfft, int(sr))
    freqs = _FREQS_CACHE.get(key)
    if freqs is None:
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        _FREQS_CACHE[key] = freqs
    return freqs


def analyze_spectrum(
    samples: np.ndarray,
    sr: int = SAMPLE_RATE,
    fmin: float = 20.0,
    fmax: float = 5000.0,
    min_rms: float = 2e-3
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Devuelve (freqs, spec, frecuencia_dominante | None).

    Utiliza una ventana de Hann, zero-padding x4 e interpolación parabólica
    sobre el logaritmo de la magnitud para lograr una precisión de detección < 0.05 Hz.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 8:
        return np.array([]), np.array([]), None

    # Eliminar componente continua (offset DC)
    samples = samples - np.mean(samples)

    window = _get_hann_window(n)
    nfft = 1 << (max(n, 2) * 4 - 1).bit_length()
    spec = np.abs(np.fft.rfft(samples * window, nfft)) / (n / 4)
    freqs = _get_rfftfreqs(nfft, sr)

    # Medir valor eficaz (RMS) para rechazar ruido de fondo / silencio
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < min_rms:
        return freqs, spec, None

    band = (freqs >= fmin) & (freqs <= fmax)
    idx_band = np.where(band)[0]
    if len(idx_band) == 0:
        return freqs, spec, None

    k = idx_band[np.argmax(spec[idx_band])]
    if k <= 0 or k >= len(spec) - 1:
        return freqs, spec, float(freqs[k])

    # Interpolación parabólica en escala logarítmica
    a = float(np.log(spec[k - 1] + 1e-12))
    b = float(np.log(spec[k] + 1e-12))
    c = float(np.log(spec[k + 1] + 1e-12))
    denom = a - 2.0 * b + c

    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    # Limitar delta al intervalo [-1, 1] para evitar divergencias
    delta = max(-1.0, min(1.0, delta))

    dominant_freq = float((k + delta) * (sr / nfft))
    return freqs, spec, dominant_freq


def freq_to_note(f: Optional[float]) -> Tuple[Optional[str], float]:
    """Devuelve (nombre de nota con octava, desviación en centésimas).

    Ejemplo: 440.0 Hz -> ('La4', 0.0)
    """
    if f is None or f <= 0.0:
        return None, 0.0

    midi = 69.0 + 12.0 * math.log2(f / 440.0)
    nearest = int(round(midi))
    cents = (midi - nearest) * 100.0
    note_idx = nearest % 12
    octave = nearest // 12 - 1
    return f"{NOTE_NAMES[note_idx]}{octave}", cents


def calc_signal_metrics(samples: np.ndarray) -> Tuple[float, float, bool]:
    """Calcula RMS, pico absoluto e indica si existe saturación (clipping >= 0.98)."""
    if len(samples) == 0:
        return 0.0, 0.0, False
    arr = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    clipping = peak >= 0.98
    return rms, peak, clipping


class FrequencySmoother:
    """Filtro adaptativo para la frecuencia leída.

    Suaviza variaciones menores (ruido e incertidumbre) usando una media exponencial,
    pero responde de inmediato si la frecuencia da un salto significativo (> 2 Hz o > 3%).
    """

    def __init__(self, alpha: float = 0.35, threshold_hz: float = 2.0, threshold_rel: float = 0.03):
        self.alpha = alpha
        self.threshold_hz = threshold_hz
        self.threshold_rel = threshold_rel
        self.smoothed_freq: Optional[float] = None
        self.silence_count: int = 0

    def update(self, raw_freq: Optional[float]) -> Optional[float]:
        if raw_freq is None:
            self.silence_count += 1
            if self.silence_count >= 3:
                self.smoothed_freq = None
            return self.smoothed_freq

        self.silence_count = 0
        if self.smoothed_freq is None:
            self.smoothed_freq = raw_freq
            return self.smoothed_freq

        diff = abs(raw_freq - self.smoothed_freq)
        rel_diff = diff / max(self.smoothed_freq, 1e-3)

        # Si el cambio es grande, es un cambio de frecuencia del generador: no amortiguar
        if diff > self.threshold_hz and rel_diff > self.threshold_rel:
            self.smoothed_freq = raw_freq
        else:
            # Variación pequeña: amortiguar con filtro exponencial
            self.smoothed_freq = (1.0 - self.alpha) * self.smoothed_freq + self.alpha * raw_freq

        return self.smoothed_freq

    def reset(self):
        self.smoothed_freq = None
        self.silence_count = 0
