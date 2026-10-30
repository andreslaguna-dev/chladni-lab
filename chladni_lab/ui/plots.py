"""Componentes gráficos de pyqtgraph para visualización espectral, osciloscopio, barrido y placa."""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from chladni_lab.physics import chladni_field, circular_chladni_field


class SpectrumPlot(pg.PlotWidget):
    """Gráfico del espectro de Fourier de la señal de entrada (micrófono o cable 3.5 mm)."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, title="Espectro en frecuencia (FFT)")
        self.setLabel("bottom", "Frecuencia", units="Hz")
        self.setLabel("left", "Magnitud normalizada")
        self.setXRange(0, 3000, padding=0.02)
        self.setYRange(0, 1.05, padding=0.02)
        self.showGrid(x=True, y=True, alpha=0.25)

        self.curve = self.plot(pen=pg.mkPen("#3388ff", width=2))
        self.peak_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff4444", width=1.5, style=QtCore.Qt.DashLine))
        self.addItem(self.peak_line)
        self.peak_line.setVisible(False)

    def update_spectrum(self, freqs: np.ndarray, spec: np.ndarray, peak: Optional[float] = None) -> None:
        if len(freqs) == 0:
            self.curve.clear()
            self.peak_line.setVisible(False)
            return

        mask = freqs <= 5000
        f_sub = freqs[mask]
        s_sub = spec[mask]

        if len(f_sub) > 2500:
            # Decimación conservando picos (max-pooling por bloque) para no perder tonos estrechos
            step = len(f_sub) // 1250
            n_blocks = len(f_sub) // step
            f_sub = f_sub[: n_blocks * step : step]
            s_sub = np.max(s_sub[: n_blocks * step].reshape(n_blocks, step), axis=1)

        max_val = float(np.max(s_sub)) if len(s_sub) > 0 else 0.0
        # Piso suave (1e-4) para que el ruido térmico en silencio no se amplifique al 100%,
        # pero cualquier silbido o tono eleve claramente el pico hasta 1.0
        denom = max(max_val, 1e-4)
        norm_spec = s_sub / denom
        self.curve.setData(f_sub, norm_spec)

        if peak is not None and peak > 0:
            self.peak_line.setValue(peak)
            self.peak_line.setVisible(True)
        else:
            self.peak_line.setVisible(False)


class OscilloscopePlot(pg.PlotWidget):
    """Osciloscopio en tiempo real con auto-escala vertical suave para micrófono y línea."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, title="Osciloscopio (señal en tiempo real)")
        self.setLabel("bottom", "Tiempo", units="ms")
        self.setLabel("left", "Amplitud")
        self.setYRange(-1.15, 1.15, padding=0.0)
        self.showGrid(x=True, y=True, alpha=0.25)

        # Líneas de referencia para advertencia de saturación
        self.clip_top = pg.InfiniteLine(pos=0.98, angle=0, pen=pg.mkPen("#cc2222", width=1, style=QtCore.Qt.DotLine))
        self.clip_bot = pg.InfiniteLine(pos=-0.98, angle=0, pen=pg.mkPen("#cc2222", width=1, style=QtCore.Qt.DotLine))
        self.addItem(self.clip_top)
        self.addItem(self.clip_bot)

        self.wave_curve = self.plot(pen=pg.mkPen("#00cc88", width=1.8))
        self._y_scale = 0.15

    def update_waveform(self, samples: np.ndarray, sr: int = 44100) -> None:
        if len(samples) == 0:
            self.wave_curve.clear()
            return

        # Mostrar aproximadamente 25 a 30 ms para una visualización clara de ondas entre 50 y 2000 Hz
        window_samples = min(len(samples), int(sr * 0.03))
        display_data = samples[-window_samples:]
        # Centrar eliminando offset DC para que ondas débiles de micrófono sean simétricas
        display_data = display_data - np.mean(display_data)
        t_ms = (np.arange(window_samples) / sr) * 1000.0

        # Auto-escala vertical suave: permite ver tanto silbidos suaves en micrófono (±0.01)
        # como señales fuertes por cable (±1.15)
        peak = float(np.max(np.abs(display_data))) if len(display_data) > 0 else 0.0
        target_scale = max(0.008, min(1.15, peak * 1.35))
        if target_scale > self._y_scale:
            self._y_scale = 0.4 * self._y_scale + 0.6 * target_scale
        else:
            self._y_scale = 0.85 * self._y_scale + 0.15 * target_scale

        self.setXRange(0, t_ms[-1], padding=0.0)
        self.setYRange(-self._y_scale, self._y_scale, padding=0.0)
        self.wave_curve.setData(t_ms, display_data)


class SweepResponsePlot(pg.PlotWidget):
    """Curva de respuesta de amplitud vs frecuencia A(f) registrada durante el barrido."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, title="Respuesta del barrido: Amplitud vs Frecuencia")
        self.setLabel("bottom", "Frecuencia", units="Hz")
        self.setLabel("left", "Amplitud de entrada")
        self.showGrid(x=True, y=True, alpha=0.25)

        self.curve = self.plot(pen=pg.mkPen("#ffaa00", width=2))
        self.peaks_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#ffffff", width=1), brush=pg.mkBrush("#ff3333"))
        self.addItem(self.peaks_scatter)
        self.cursor_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#00eeff", width=1.5))
        self.addItem(self.cursor_line)
        self.cursor_line.setVisible(False)

    def clear_sweep(self) -> None:
        self.curve.clear()
        self.peaks_scatter.clear()
        self.cursor_line.setVisible(False)

    def update_data(self, freqs: List[float], amps: List[float], current_f: Optional[float] = None) -> None:
        if len(freqs) == 0:
            return
        f_arr = np.array(freqs)
        a_arr = np.array(amps)
        self.curve.setData(f_arr, a_arr)
        if current_f is not None:
            self.cursor_line.setValue(current_f)
            self.cursor_line.setVisible(True)

    def set_detected_peaks(self, peak_freqs: List[float], peak_amps: List[float]) -> None:
        if len(peak_freqs) > 0:
            self.peaks_scatter.setData(peak_freqs, peak_amps)
        else:
            self.peaks_scatter.clear()


class PlatePlot(pg.PlotWidget):
    """Visualizador 2D interactivo de la placa de Chladni (cuadrada o circular)."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, title="Simulación de la placa de Chladni")
        self.setAspectLocked(True)
        self.setXRange(-0.02, 1.02, padding=0.0)
        self.setYRange(-0.02, 1.02, padding=0.0)
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.setBackground("#15171c")

        self.heat_img = pg.ImageItem()
        self.addItem(self.heat_img)

        # Contorno visual de la placa
        self.border_item = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
        self.border_item.setPen(QtGui.QPen(QtGui.QColor(90, 95, 110), 2))
        self.addItem(self.border_item)

        # Círculo auxiliar para placa circular
        self.circle_border = QtWidgets.QGraphicsEllipseItem(0, 0, 1, 1)
        self.circle_border.setPen(QtGui.QPen(QtGui.QColor(110, 140, 190), 2))
        self.addItem(self.circle_border)
        self.circle_border.setVisible(False)

        # Partículas de arena
        self.sand_scatter = pg.ScatterPlotItem(
            size=2.2,
            pen=None,
            brush=pg.mkBrush(240, 240, 245, 230),
            pxMode=True
        )
        self.addItem(self.sand_scatter)

        self._cached_key: Optional[Tuple] = None
        self._geometry = "square"

    def set_geometry(self, geometry: str) -> None:
        self._geometry = geometry
        if geometry == "circular":
            self.border_item.setVisible(False)
            self.circle_border.setVisible(True)
        else:
            self.border_item.setVisible(True)
            self.circle_border.setVisible(False)
        self._cached_key = None

    def update_nodal_lines(
        self,
        n: int,
        m: int,
        sign: int = 1,
        visible: bool = True
    ) -> None:
        self.heat_img.setVisible(visible)
        if not visible:
            return

        key = (self._geometry, n, m, sign)
        if self._cached_key == key:
            return

        res = 160
        grid = np.linspace(0, 1, res)
        xx, yy = np.meshgrid(grid, grid, indexing="ij")

        if self._geometry == "circular":
            dx = xx - 0.5
            dy = yy - 0.5
            r = np.sqrt(dx * dx + dy * dy)
            r_norm = np.clip(r / 0.5, 0.0, 1.0)
            theta = np.arctan2(dy, dx)
            field = np.abs(circular_chladni_field(n, m, r_norm, theta))
            # Ocultar fuera del radio 0.5
            field[r > 0.5] = 1.0
        else:
            field = np.abs(chladni_field(n, m, xx, yy, sign=sign))

        max_val = np.max(field) or 1.0
        norm_field = field / max_val

        # Fondo sutil: azul oscuro translúcido donde no hay vibración (líneas nodales)
        rgba = np.zeros(norm_field.shape + (4,), dtype=np.uint8)
        rgba[..., 0] = 50
        rgba[..., 1] = 85
        rgba[..., 2] = 160
        # Más opaco donde la placa vibra, transparente en las líneas nodales
        rgba[..., 3] = (np.clip(norm_field, 0.0, 1.0) ** 0.7 * 115).astype(np.uint8)

        if self._geometry == "circular":
            dx = xx - 0.5
            dy = yy - 0.5
            rgba[dx * dx + dy * dy > 0.25, 3] = 0

        self.heat_img.setImage(rgba, autoLevels=False)
        self.heat_img.setRect(QtCore.QRectF(0, 0, 1, 1))
        self._cached_key = key

    def update_sand(self, xs: np.ndarray, ys: np.ndarray) -> None:
        pos = np.column_stack((xs, ys))
        self.sand_scatter.setData(pos=pos)

    def grab_screenshot(self, filepath: Path) -> bool:
        """Captura la imagen de la placa simulada y la guarda en formato PNG."""
        try:
            pixmap = self.grab()
            return pixmap.save(str(filepath), "PNG")
        except Exception:
            return False
