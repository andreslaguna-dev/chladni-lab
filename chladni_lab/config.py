"""Gestión de configuración persistente en ~/.config/chladni_lab/config.json."""
import json
import logging
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path.home() / ".config" / "chladni_lab"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "output_device": None,
    "input_device": None,
    "sample_rate": 44100,
    "frequency": 200.0,
    "volume": 0.15,
    "f0": 100.0,
    "exponent": 1.0,
    "theme": "dark",
    "plate_type": "square",
    "grain_count": 7000,
    "grain_speed": 0.04,
    "sign": 1,
}

FALLBACK_CONFIG_FILE = Path(".chladni_config.json")

logger = logging.getLogger("chladni_lab.config")


def _get_active_config_path() -> Path:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return CONFIG_FILE
    except Exception:
        return FALLBACK_CONFIG_FILE


def load_config() -> Dict[str, Any]:
    """Carga la configuración del usuario. Si el archivo no existe o está dañado, usa valores por defecto."""
    cfg = dict(DEFAULT_CONFIG)
    target = _get_active_config_path()
    if not target.exists():
        return cfg

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update(data)
        else:
            logger.warning("El archivo de configuración no contiene un objeto JSON válido. Usando valores por defecto.")
    except Exception as exc:
        logger.warning(f"Error al leer configuración ({target}): {exc}. Usando valores por defecto.")

    return cfg


def save_config(cfg: Dict[str, Any]) -> bool:
    """Guarda la configuración en disco de forma segura."""
    target = _get_active_config_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        full_cfg = dict(DEFAULT_CONFIG)
        full_cfg.update(cfg)
        target.write_text(json.dumps(full_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as exc:
        logger.error(f"Error al guardar configuración ({target}): {exc}")
        return False

