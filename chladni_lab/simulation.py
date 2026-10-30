"""Simulación física vectorizada de partículas de arena/sal sobre placas de Chladni.

Implementado en NumPy vectorizado de alto rendimiento para garantizar >= 25 FPS
con 7000 granos tanto en placas cuadradas como circulares.
"""
from typing import Optional, Tuple
import numpy as np

from chladni_lab.physics import chladni_field, circular_chladni_field


class SandSimulation:
    """Simulador de granos de arena sujetos a la vibración superficial de la placa.

    Donde la placa vibra fuertemente (antinodos), los granos sufren mayor aceleración
    y rebotan con mayor dispersión. En las líneas nodales (amplitud cero), la fuerza
    neta de vibración es mínima, por lo que los granos se depositan naturalmente allí.
    """

    def __init__(self, count: int = 7000, seed: Optional[int] = None, geometry: str = "square"):
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.geometry = geometry
        self.x = np.zeros(self.count, dtype=np.float64)
        self.y = np.zeros(self.count, dtype=np.float64)
        self.reset()

    def set_count(self, count: int) -> None:
        """Modifica la cantidad de granos de arena y reinicia su posición."""
        self.count = max(500, min(30000, int(count)))
        self.reset()

    def set_geometry(self, geometry: str) -> None:
        """Cambia entre 'square' (cuadrada) y 'circular' (circular)."""
        if geometry in ("square", "circular"):
            self.geometry = geometry
            self.reset()

    def reset(self) -> None:
        """Distribuye uniformemente los granos sobre la geometría de la placa."""
        if self.geometry == "circular":
            # Distribución uniforme dentro de un círculo de radio R = 0.5 centrado en (0.5, 0.5)
            r = 0.5 * np.sqrt(self.rng.random(self.count))
            theta = self.rng.uniform(-np.pi, np.pi, self.count)
            self.x = 0.5 + r * np.cos(theta)
            self.y = 0.5 + r * np.sin(theta)
        else:
            # Distribución uniforme en el cuadrado [0, 1] x [0, 1]
            self.x = self.rng.random(self.count)
            self.y = self.rng.random(self.count)

    def step(
        self,
        n: int,
        m: int,
        strength: float,
        jitter: float = 0.04,
        grain: float = 0.06,
        sign: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Avanza un paso de simulación.

        Retorna (x, y) con las posiciones actualizadas de todos los granos.
        """
        if strength <= 0.0:
            return self.x, self.y

        if self.geometry == "circular":
            # Coordenadas relativas al centro (0.5, 0.5)
            dx = self.x - 0.5
            dy = self.y - 0.5
            r = np.hypot(dx, dy)
            # Normalizar r a [0, 1] multiplicando por 2.0 (1 / 0.5)
            r_norm = np.clip(r * 2.0, 0.0, 1.0)
            theta = np.arctan2(dy, dx)

            amp = np.abs(circular_chladni_field(n, m, r_norm, theta)) + grain
            step_size = jitter * strength * amp

            # Salto estocástico
            self.x += self.rng.standard_normal(self.count) * step_size
            self.y += self.rng.standard_normal(self.count) * step_size

            # Rebote elástico en el borde circular (radio = 0.5, umbral 0.495^2 = 0.245025)
            new_dx = self.x - 0.5
            new_dy = self.y - 0.5
            new_r_sq = new_dx * new_dx + new_dy * new_dy
            outside = new_r_sq > 0.245025
            if np.any(outside):
                # Reflejar de vuelta radialmente calculando sqrt solo en partículas fuera del borde
                out_r = np.sqrt(new_r_sq[outside])
                scale = (0.495 * 2.0 - out_r) / np.maximum(out_r, 1e-6)
                scale = np.clip(scale, 0.1, 0.495)
                self.x[outside] = 0.5 + new_dx[outside] * scale
                self.y[outside] = 0.5 + new_dy[outside] * scale

        else:
            # Placa cuadrada
            amp = np.abs(chladni_field(n, m, self.x, self.y, sign=sign)) + grain
            step_size = jitter * strength * amp

            self.x += self.rng.standard_normal(self.count) * step_size
            self.y += self.rng.standard_normal(self.count) * step_size

            # Rebote en los bordes del cuadrado [0, 1] in-place sin reasignar arreglos
            np.abs(self.x, out=self.x)
            np.abs(self.y, out=self.y)
            over_x = self.x > 1.0
            if np.any(over_x):
                self.x[over_x] = 2.0 - self.x[over_x]
            over_y = self.y > 1.0
            if np.any(over_y):
                self.y[over_y] = 2.0 - self.y[over_y]

            # Recorte de seguridad
            np.clip(self.x, 0.0, 1.0, out=self.x)
            np.clip(self.y, 0.0, 1.0, out=self.y)

        return self.x, self.y
