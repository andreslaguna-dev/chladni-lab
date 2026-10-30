"""Laboratorio de figuras de Chladni.

Uso:
    python chladni_lab.py
    python -m chladni_lab
"""
import sys
from pathlib import Path

# Asegurar que el directorio del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chladni_lab.__main__ import main

if __name__ == "__main__":
    main()
