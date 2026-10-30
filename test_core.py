import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from chladni_core import (SAMPLE_RATE, analyze_spectrum, freq_to_note, best_mode,
                          mode_table, chladni_field, SandSimulation)

t = np.arange(8192) / SAMPLE_RATE

# 1) detección de frecuencia con varias frecuencias de prueba
for f in (82.41, 110.0, 261.63, 440.0, 1234.5, 3000.0):
    sig = 0.5 * np.sin(2 * np.pi * f * t) + 0.01 * np.random.randn(len(t))
    _, _, est = analyze_spectrum(sig)
    print(f"real {f:8.2f} Hz -> detectada {est:8.2f} Hz  (error {abs(est - f):.3f} Hz)")
    assert abs(est - f) < 0.5, "error de detección demasiado grande"

# 2) silencio no debe dar frecuencia
_, _, est = analyze_spectrum(np.zeros(8192))
assert est is None
print("silencio ->", est)

# 3) notas
print(freq_to_note(440.0), freq_to_note(82.41), freq_to_note(445.0))
assert freq_to_note(440.0)[0] == "La4"

# 4) modos de Chladni: simetría y nodos
x = np.linspace(0, 1, 11)
xx, yy = np.meshgrid(x, x, indexing="ij")
u = chladni_field(1, 2, xx, yy)
assert np.allclose(u, u.T), "el campo simétrico debe ser igual al intercambiar x e y"
ua = chladni_field(1, 2, xx, yy, sign=-1)
assert np.allclose(ua, -ua.T), "el campo antisimétrico cambia de signo"
assert np.allclose(chladni_field(1, 2, xx, xx, sign=-1), 0), "la diagonal es nodal en el antisimétrico"
print("campos (1,2) correctos: simétrico y antisimétrico")

# 5) mejor modo y resonancia
f0 = 100.0
n, m, fm, s = best_mode(500.0, f0)
print("modo para 500 Hz:", n, m, fm, round(s, 3))
assert (n, m) == (1, 2) and abs(fm - 500) < 1e-9 and s > 0.99
_, _, _, s_off = best_mode(540.0, f0)
print("resonancia fuera de pico (540 Hz):", round(s_off, 3))
assert s_off < 0.3

# 6) la arena debe migrar a las líneas nodales
sim = SandSimulation(count=6000, seed=1)
before = np.mean(np.abs(chladni_field(2, 5, sim.x, sim.y)))
for _ in range(600):
    sim.step(2, 5, 1.0)
after = np.mean(np.abs(chladni_field(2, 5, sim.x, sim.y)))
print(f"vibración media bajo la arena: antes {before:.3f} -> después {after:.3f}")
assert after < before * 0.5, (before, after)
assert sim.x.min() >= 0 and sim.x.max() <= 1 and sim.y.min() >= 0 and sim.y.max() <= 1

# 7) sin resonancia la arena casi no se mueve
sim2 = SandSimulation(count=2000, seed=2)
x0 = sim2.x.copy()
for _ in range(100):
    sim2.step(2, 5, 0.0)
assert np.allclose(x0, sim2.x)
print("sin resonancia la arena no se mueve")
print("TODAS LAS PRUEBAS PASARON")
