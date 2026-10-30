"""Física de figuras de Chladni para placas cuadradas y circulares.

Contiene las funciones analíticas del campo de desplazamiento vertical,
cálculo de frecuencias teóricas, búsqueda del modo resonante más cercano
y asistente de calibración por mínimos cuadrados.
Sin dependencias de Qt.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.special as sp


# --------------------------------------------------------------------------
# Placa cuadrada (aproximación clásica de Chladni / Ritz)
# --------------------------------------------------------------------------
def chladni_field(
    n: int,
    m: int,
    x: np.ndarray,
    y: np.ndarray,
    sign: int = 1
) -> np.ndarray:
    """Desplazamiento vertical de un modo (n, m) en placa cuadrada, x e y en [0, 1].

    sign = +1: Modos simétricos (habituales con excitación en el centro).
    sign = -1: Modos antisimétricos (tienen siempre la diagonal x = y como línea nodal).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    return (
        np.cos(n * np.pi * x_arr) * np.cos(m * np.pi * y_arr)
        + sign * np.cos(m * np.pi * x_arr) * np.cos(n * np.pi * y_arr)
    )


def square_mode_factor(n: int, m: int) -> float:
    """Factor geométrico característico para placa cuadrada (n² + m²)."""
    return float(n * n + m * m)


def mode_table(f0: float, max_m: int = 14, exponent: float = 1.0) -> List[Tuple[int, int, float]]:
    """Tabla ordenada de modos (n, m, frecuencia_teorica) con n < m.

    Modelo generalizado: f(n, m) = f0 * (n² + m²)^p.
    """
    modes = []
    for m in range(1, max_m + 1):
        for n in range(0, m):
            factor = square_mode_factor(n, m)
            freq = f0 * (factor ** exponent)
            modes.append((n, m, freq))
    modes.sort(key=lambda t: t[2])
    return modes


def best_mode(
    freq: float,
    f0: float,
    bandwidth_frac: float = 0.015,
    max_m: int = 14,
    exponent: float = 1.0
) -> Tuple[int, int, float, float]:
    """Encuentra el modo cuadrado más cercano y calcula su fuerza de resonancia (0.0 a 1.0)."""
    modes = mode_table(f0, max_m=max_m, exponent=exponent)
    if not modes:
        return 0, 1, f0, 0.0
    n, m, fm = min(modes, key=lambda t: abs(t[2] - freq))
    bw = max(fm * bandwidth_frac, 1e-6)
    strength = 1.0 / (1.0 + ((freq - fm) / bw) ** 2)
    return n, m, fm, strength


# --------------------------------------------------------------------------
# Placa circular (funciones de Bessel ordinarias)
# --------------------------------------------------------------------------
# Memoria caché de ceros de Bessel para eficiencia en tiempo real
_BESSEL_ZEROS_CACHE: Dict[Tuple[int, int], float] = {}


def get_bessel_zero(m: int, n: int) -> float:
    """Obtiene el n-ésimo cero positivo (n >= 1) de la función de Bessel J_m."""
    key = (m, n)
    if key not in _BESSEL_ZEROS_CACHE:
        zeros = sp.jn_zeros(m, n)
        _BESSEL_ZEROS_CACHE[key] = float(zeros[-1])
    return _BESSEL_ZEROS_CACHE[key]


def circular_chladni_field(
    m: int,
    n: int,
    r: np.ndarray,
    theta: np.ndarray
) -> np.ndarray:
    """Desplazamiento vertical u(r, theta) para placa circular, r in [0, 1], theta in [-pi, pi].

    m: número de diámetros nodales (simetría angular cos(m * theta)).
    n: número de círculos nodales (n >= 1).

    Aproximación física:
    En una placa circular delgada homogénea con sujeción central y borde libre,
    los modos vibracionales se expresan en función de coordenadas polares (r, theta)
    mediante combinaciones de funciones de Bessel ordinarias J_m(k*r) y modificadas I_m(k*r).
    En este modelo pedagógico empleamos la aproximación clásica de Chladni-Rayleigh
    basada en los ceros de J_m(alpha_{m,n} * r) * cos(m * theta), que reproduce fielmente
    las familias de patrones nodales concéntricos y radiales observados en laboratorio.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    th_arr = np.asarray(theta, dtype=np.float64)

    # Validar que n >= 1 (al menos un círculo nodal o borde)
    n_safe = max(1, n)
    alpha = get_bessel_zero(m, n_safe)
    radial = sp.jv(m, alpha * np.clip(r_arr, 0.0, 1.0))
    angular = np.cos(m * th_arr)
    return radial * angular


def circular_mode_table(
    f0: float,
    max_diameters: int = 5,
    max_circles: int = 4,
    exponent: float = 1.0
) -> List[Tuple[int, int, float]]:
    """Tabla de modos para placa circular (m_diametros, n_circulos, frecuencia_teorica)."""
    modes = []
    base_alpha = get_bessel_zero(0, 1)  # Cero de referencia fundamental J0,1 = 2.4048
    for m in range(0, max_diameters + 1):
        for n in range(1, max_circles + 1):
            alpha = get_bessel_zero(m, n)
            # La frecuencia en placas delgadas es proporcional a alpha^2
            factor = (alpha / base_alpha) ** 2
            freq = f0 * (factor ** exponent)
            modes.append((m, n, freq))
    modes.sort(key=lambda t: t[2])
    return modes


def best_circular_mode(
    freq: float,
    f0: float,
    bandwidth_frac: float = 0.015,
    max_diameters: int = 5,
    max_circles: int = 4,
    exponent: float = 1.0
) -> Tuple[int, int, float, float]:
    """Modo circular más cercano a freq y fuerza de resonancia (0.0 a 1.0)."""
    modes = circular_mode_table(f0, max_diameters=max_diameters, max_circles=max_circles, exponent=exponent)
    if not modes:
        return 0, 1, f0, 0.0
    m, n, fm = min(modes, key=lambda t: abs(t[2] - freq))
    bw = max(fm * bandwidth_frac, 1e-6)
    strength = 1.0 / (1.0 + ((freq - fm) / bw) ** 2)
    return m, n, fm, strength


# --------------------------------------------------------------------------
# Asistente de calibración por mínimos cuadrados
# --------------------------------------------------------------------------
def calibrate_model(
    observations: List[Tuple[float, int, int]],
    geometry: str = "square"
) -> Tuple[float, float, float]:
    """Ajusta f0 y el exponente p a partir de pares (frecuencia_medida, n, m).

    Modelo linealizado en logaritmos:
        f = f0 * K^p  ==>  log(f) = log(f0) + p * log(K)

    Devuelve (f0_estimado, p_estimado, r_cuadrado).
    Si hay menos de 2 observaciones válidas, devuelve (f0_promedio, 1.0, 1.0).
    """
    if len(observations) < 2:
        if len(observations) == 1:
            f_obs, n, m = observations[0]
            if geometry == "square":
                k = square_mode_factor(n, m)
            else:
                base = get_bessel_zero(0, 1)
                k = (get_bessel_zero(n, max(1, m)) / base) ** 2
            return float(f_obs / max(k, 1e-6)), 1.0, 1.0
        return 100.0, 1.0, 0.0

    k_vals = []
    f_vals = []

    base_alpha = get_bessel_zero(0, 1)
    for f_obs, n, m in observations:
        if f_obs <= 0:
            continue
        if geometry == "square":
            k = square_mode_factor(n, m)
        else:
            k = (get_bessel_zero(n, max(1, m)) / base_alpha) ** 2

        if k > 0:
            k_vals.append(k)
            f_vals.append(f_obs)

    if len(k_vals) < 2:
        return 100.0, 1.0, 0.0

    log_k = np.log(k_vals)
    log_f = np.log(f_vals)

    # Regresión lineal: log_f = p * log_k + log_f0
    coeffs, residuals, _, _, _ = np.polyfit(log_k, log_f, deg=1, full=True)
    p = float(coeffs[0])
    log_f0 = float(coeffs[1])
    f0 = float(np.exp(log_f0))

    # Coeficiente de determinación R²
    y_pred = p * log_k + log_f0
    ss_tot = float(np.sum((log_f - np.mean(log_f)) ** 2))
    ss_res = float(np.sum((log_f - y_pred) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    r2 = max(0.0, min(1.0, r2))

    return f0, p, r2

