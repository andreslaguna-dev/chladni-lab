"""Configuración de registro (logging) para el laboratorio de Chladni."""
import logging
from pathlib import Path
import sys

LOG_DIR = Path.home() / ".local" / "share" / "chladni_lab"
LOG_FILE = LOG_DIR / "chladni_lab.log"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Inicializa el sistema de logging escribiendo en ~/.local/share/chladni_lab/ o en el directorio local."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    target_dir = LOG_DIR
    target_file = LOG_FILE
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback si ~/.local es de solo lectura (por ejemplo en entornos restringidos)
        target_dir = Path(".chladni_lab_logs")
        target_file = target_dir / "chladni_lab.log"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            target_file = None

    logger = logging.getLogger("chladni_lab")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Evita duplicar manejadores si se llama múltiples veces
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        if target_file is not None:
            try:
                file_handler = logging.FileHandler(target_file, encoding="utf-8")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

