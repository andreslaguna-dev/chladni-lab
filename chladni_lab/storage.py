"""Módulo de persistencia y exportación de registros experimentales.

Guarda y recupera resonancias medidas y comparaciones con afinadores de guitarra en JSON,
y permite exportar los datos a formato CSV.
Sin dependencias de Qt.
"""
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("chladni_lab.storage")

DEFAULT_SAVE_FILE = Path("resonancias.json")


class ExperimentRecord:
    """Representa una fila del registro experimental de Chladni."""

    def __init__(
        self,
        freq_prog: float,
        freq_tuner: Optional[float] = None,
        note: str = "",
        mode_observed: str = "",
        photo_path: str = "",
        timestamp: Optional[str] = None
    ):
        self.freq_prog = float(round(freq_prog, 2))
        self.freq_tuner = float(round(freq_tuner, 2)) if freq_tuner is not None and freq_tuner > 0 else None
        self.note = str(note)
        self.mode_observed = str(mode_observed)
        self.photo_path = str(photo_path)
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def error_rel(self) -> Optional[float]:
        """Error relativo porcentual: |f_afinador - f_programa| / f_programa * 100."""
        if self.freq_tuner is not None and self.freq_prog > 0:
            return float(abs(self.freq_tuner - self.freq_prog) / self.freq_prog * 100.0)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freq_prog": self.freq_prog,
            "freq_tuner": self.freq_tuner,
            "note": self.note,
            "error_rel": round(self.error_rel, 3) if self.error_rel is not None else None,
            "mode_observed": self.mode_observed,
            "photo_path": self.photo_path,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        # Compatibilidad con formato antiguo {"frecuencia": ..., "nota": ...}
        freq_prog = d.get("freq_prog", d.get("frecuencia", 0.0))
        freq_tuner = d.get("freq_tuner", None)
        note = d.get("note", d.get("nota", ""))
        mode_obs = d.get("mode_observed", "")
        photo = d.get("photo_path", "")
        ts = d.get("timestamp", None)
        return cls(
            freq_prog=float(freq_prog),
            freq_tuner=float(freq_tuner) if freq_tuner is not None else None,
            note=str(note),
            mode_observed=str(mode_obs),
            photo_path=str(photo),
            timestamp=ts,
        )


class ExperimentStorage:
    """Administrador de persistencia de experimentos en JSON y CSV."""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or DEFAULT_SAVE_FILE
        self.records: List[ExperimentRecord] = []
        self.load()

    def load(self, path: Optional[Path] = None) -> List[ExperimentRecord]:
        target = path or self.file_path
        self.records = []
        if not target.exists():
            return self.records

        try:
            content = target.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        try:
                            self.records.append(ExperimentRecord.from_dict(item))
                        except Exception as parse_err:
                            logger.warning(f"Error interpretando registro {item}: {parse_err}")
            logger.info(f"Cargados {len(self.records)} registros desde {target}")
        except Exception as exc:
            logger.error(f"Error al cargar archivo de experimentos ({target}): {exc}. Iniciando lista vacía.")

        return self.records

    def save(self, path: Optional[Path] = None) -> bool:
        target = path or self.file_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = [r.to_dict() for r in self.records]
            target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Guardados {len(self.records)} registros en {target}")
            return True
        except Exception as exc:
            logger.error(f"Error al guardar experimentos ({target}): {exc}")
            return False

    def add(self, record: ExperimentRecord) -> None:
        self.records.append(record)
        self.save()

    def remove(self, index: int) -> bool:
        if 0 <= index < len(self.records):
            self.records.pop(index)
            self.save()
            return True
        return False

    def export_csv(self, export_path: Path) -> bool:
        """Exporta los registros a un archivo CSV."""
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Frecuencia_Programa_Hz",
                    "Frecuencia_Afinador_Hz",
                    "Nota",
                    "Error_Relativo_Pct",
                    "Modo_Observado",
                    "Ruta_Foto",
                    "Fecha"
                ])
                for r in self.records:
                    writer.writerow([
                        f"{r.freq_prog:.2f}",
                        f"{r.freq_tuner:.2f}" if r.freq_tuner is not None else "",
                        r.note,
                        f"{r.error_rel:.3f}" if r.error_rel is not None else "",
                        r.mode_observed,
                        r.photo_path,
                        r.timestamp,
                    ])
            logger.info(f"Exportados {len(self.records)} registros a CSV: {export_path}")
            return True
        except Exception as exc:
            logger.error(f"Error al exportar CSV a {export_path}: {exc}")
            return False

