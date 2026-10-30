"""Punto de entrada para el Laboratorio de Chladni.

Uso:
    python -m chladni_lab             (Abre la interfaz gráfica)
    python -m chladni_lab --selftest  (Ejecuta autodiagnóstico de audio)
"""
import argparse
import sys
import time
import numpy as np

from chladni_lab.config import load_config
from chladni_lab.dsp import analyze_spectrum
from chladni_lab.logging_setup import setup_logging


def run_selftest(out_dev=None, in_dev=None, duration=1.5, test_freq=440.0) -> int:
    """Ejecuta un autodiagnóstico en bucle cerrado (loopback) o con generador sintético."""
    print("=" * 60)
    print("AUTODIAGNÓSTICO: LABORATORIO DE CHLADNI")
    print("=" * 60)
    print(f"Frecuencia objetivo del generador: {test_freq:.2f} Hz")

    from chladni_lab.audio import ToneGenerator, InputMonitor, list_audio_devices

    out_devs = list_audio_devices("output")
    in_devs = list_audio_devices("input")
    print(f"Dispositivos detectados: {len(out_devs)} salidas, {len(in_devs)} entradas.")

    tone = ToneGenerator(sr=44100)
    tone.frequency = test_freq
    tone.volume = 0.35
    tone.playing = True

    monitor = InputMonitor(sr=44100, seconds=1.0)

    print("Iniciando generación y captura...")
    ok_out, msg_out = tone.start(device=out_dev)
    ok_in, msg_in = monitor.start(device=in_dev)

    if not ok_out or not ok_in:
        print(f"AVISO: No se pudo abrir audio en hardware real (salida: {msg_out}, entrada: {msg_in}).")
        print("Probando análisis con señal interna sintetizada...")
        sr = 44100
        t = np.arange(int(sr * 0.5)) / sr
        synth_signal = 0.5 * np.sin(2 * np.pi * test_freq * t) + 0.005 * np.random.randn(len(t))
        freqs, spec, detected = analyze_spectrum(synth_signal, sr=sr)
        err = abs(detected - test_freq) if detected else 999.0
        print(f"Frecuencia detectada en simulación: {detected:.3f} Hz (error: {err:.4f} Hz)")
        assert err < 0.05, f"Error excesivo en análisis: {err}"
        print("✔ Autodiagnóstico del procesador DSP: APROBADO (error < 0.05 Hz).")
        return 0

    try:
        # Dar tiempo a que el búfer se llene con el audio real
        time.sleep(duration)
        samples = monitor.snapshot(8192)
        sr = monitor.sr or 44100
        freqs, spec, detected = analyze_spectrum(samples, sr=sr)

        print("\n--- Resultados de Captura por Cable 3.5 mm ---")
        if detected is not None:
            err = abs(detected - test_freq)
            print(f"Frecuencia detectada: {detected:.2f} Hz")
            print(f"Error absoluto:       {err:.3f} Hz ({err / test_freq * 100:.2f} %)")
            if err < 2.0:
                print("✔ Bucle de audio verificado con éxito: EXCELENTE.")
            else:
                print("⚠ Señal detectada pero con desviación. Verifique volumen y tarjeta.")
        else:
            print("⚠ Sin señal detectable en la entrada.")
            print("  Verifique que el cable 3.5 mm conecte la salida de audio con la entrada de línea/micrófono,")
            print("  y que el dispositivo de entrada tenga volumen activo (en Configuración de Sonido de Windows,")
            print("  o en pavucontrol / alsamixer en Linux).")
    finally:
        tone.stop()
        monitor.stop()

    print("=" * 60)
    return 0


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="Laboratorio de Figuras de Chladni")
    parser.add_argument("--selftest", action="store_true", help="Ejecutar autodiagnóstico de audio")
    parser.add_argument("--theme", choices=["dark", "light"], default=None, help="Tema de la interfaz")
    args = parser.parse_args()

    setup_logging()

    if args.selftest:
        sys.exit(run_selftest())

    from PyQt5 import QtWidgets
    from chladni_lab.ui.main_window import MainWindow
    from chladni_lab.ui.theme import apply_theme, get_stylesheet

    app = QtWidgets.QApplication(sys.argv)
    cfg = load_config()
    theme = args.theme or cfg.get("theme", "dark")
    apply_theme(app, theme)
    app.setStyleSheet(get_stylesheet(theme))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

