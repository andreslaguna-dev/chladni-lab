"""Núcleo del laboratorio de Chladni: audio, detección de frecuencia y simulación.

Módulo de compatibilidad hacia atrás que re-exporta las implementaciones
optimizadas del paquete chladni_lab (audio, dsp, physics, simulation).
Sin dependencias de interfaz gráfica, apto para pruebas independientes.
"""
from chladni_lab.audio import SAMPLE_RATE, InputMonitor, ToneGenerator
from chladni_lab.dsp import NOTE_NAMES, analyze_spectrum, freq_to_note
from chladni_lab.physics import (best_circular_mode, best_mode, chladni_field,
                                circular_chladni_field, circular_mode_table,
                                mode_table)
from chladni_lab.simulation import SandSimulation

__all__ = [
    "SAMPLE_RATE",
    "NOTE_NAMES",
    "analyze_spectrum",
    "freq_to_note",
    "ToneGenerator",
    "InputMonitor",
    "chladni_field",
    "circular_chladni_field",
    "mode_table",
    "circular_mode_table",
    "best_mode",
    "best_circular_mode",
    "SandSimulation",
]
