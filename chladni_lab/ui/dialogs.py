"""Diálogos interactivos: Asistente de calibración, confirmación de resonancias y edición de registros."""
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets

from chladni_lab.physics import calibrate_model
from chladni_lab.storage import ExperimentRecord


class CalibrationDialog(QtWidgets.QDialog):
    """Asistente de calibración por mínimos cuadrados a partir de frecuencias reales medidas."""

    def __init__(self, current_f0: float, current_p: float, geometry: str = "square", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asistente de calibración de placa (Mínimos Cuadrados)")
        self.resize(550, 420)
        self.geometry_type = geometry
        self.calibrated_f0 = current_f0
        self.calibrated_p = current_p
        self.applied = False

        self._build_ui(current_f0, current_p)

    def _build_ui(self, f0: float, p: float):
        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "<b>Ajuste del modelo físico f = f0 · (K)<sup>p</sup></b><br>"
            "Ingresa 2 o más observaciones (frecuencia medida en Hz y modo observado n, m) "
            "para estimar f0 y el exponente característico de tu placa real."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Frecuencia medida (Hz)", "Modo n", "Modo m"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)

        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ Agregar fila")
        del_btn = QtWidgets.QPushButton("- Eliminar fila")
        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._del_row)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        layout.addLayout(btn_row)

        self.result_label = QtWidgets.QLabel(f"Valores actuales: f0 = {f0:.2f} Hz, exponente p = {p:.3f}")
        self.result_label.setStyleSheet("font-weight: bold; color: #4a90e2; padding: 6px;")
        layout.addWidget(self.result_label)

        calc_btn = QtWidgets.QPushButton("⚡ Calcular ajuste por mínimos cuadrados")
        calc_btn.clicked.connect(self._compute_fit)
        layout.addWidget(calc_btn)

        action_row = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton("✔ Aplicar a la simulación")
        apply_btn.clicked.connect(self._apply)
        cancel_btn = QtWidgets.QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        action_row.addStretch()
        action_row.addWidget(apply_btn)
        action_row.addWidget(cancel_btn)
        layout.addLayout(action_row)

        # Añadir filas de ejemplo por defecto
        self._add_row(f0 * 5.0, 1, 2)
        self._add_row(f0 * 13.0, 2, 3)

    def _add_row(self, freq: float = 500.0, n: int = 1, m: int = 2):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{freq:.2f}"))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(n)))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(m)))

    def _del_row(self):
        curr = self.table.currentRow()
        if curr >= 0:
            self.table.removeRow(curr)

    def _compute_fit(self):
        obs = []
        for r in range(self.table.rowCount()):
            try:
                f_val = float(self.table.item(r, 0).text())
                n_val = int(self.table.item(r, 1).text())
                m_val = int(self.table.item(r, 2).text())
                if f_val > 0:
                    obs.append((f_val, n_val, m_val))
            except Exception:
                continue

        if len(obs) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Datos insuficientes", "Se necesitan al menos 2 observaciones válidas para realizar la regresión."
            )
            return

        f0_fit, p_fit, r2 = calibrate_model(obs, geometry=self.geometry_type)
        self.calibrated_f0 = f0_fit
        self.calibrated_p = p_fit
        self.result_label.setText(
            f"Resultado: f0 = {f0_fit:.2f} Hz | exponente p = {p_fit:.4f} | R² = {r2:.4f}"
        )

    def _apply(self):
        self.applied = True
        self.accept()


class ResonanceSuggestionDialog(QtWidgets.QDialog):
    """Diálogo pedagógico que presenta los picos detectados durante el barrido."""

    def __init__(self, peaks: List[Tuple[float, float]], on_save_callback: Callable, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Picos de resonancia detectados en el barrido")
        self.resize(580, 400)
        self.peaks = peaks
        self.on_save_callback = on_save_callback

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        disclaimer = QtWidgets.QLabel(
            "<b>⚠️ Nota física esencial sobre la medición:</b><br>"
            "Cuando se utiliza el cable 1 a 1 de retorno a la entrada de audio, la curva registrada "
            "refleja la <i>respuesta eléctrica</i> del sistema y de la tarjeta, <b>no</b> la vibración mecánica "
            "de la placa metálica.<br><br>"
            "Para medir las resonancias mecánicas reales de la placa, conecta un micrófono o sensor piezoeléctrico "
            "a la entrada de audio y excita la placa con el parlante.<br>"
            "Selecciona a continuación los picos que deseas confirmar para guardarlos en el registro de experimentos:"
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("background-color: #3b321a; color: #ffdd88; padding: 10px; border-radius: 5px;")
        layout.addWidget(disclaimer)

        self.table = QtWidgets.QTableWidget(len(self.peaks), 3)
        self.table.setHorizontalHeaderLabels(["Guardar", "Frecuencia pico (Hz)", "Amplitud relativa"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        for i, (f_peak, amp) in enumerate(self.peaks):
            chk = QtWidgets.QCheckBox()
            chk.setChecked(True)
            chk_cell = QtWidgets.QWidget()
            chk_layout = QtWidgets.QHBoxLayout(chk_cell)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(QtCore.Qt.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)

            self.table.setCellWidget(i, 0, chk_cell)
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{f_peak:.2f}"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{amp:.3f}"))

        layout.addWidget(self.table)

        btn_row = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("✔ Guardar seleccionadas en registro")
        save_btn.clicked.connect(self._save_selected)
        close_btn = QtWidgets.QPushButton("Cerrar")
        close_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _save_selected(self):
        saved_count = 0
        for i in range(self.table.rowCount()):
            cell_widget = self.table.cellWidget(i, 0)
            chk = cell_widget.findChild(QtWidgets.QCheckBox)
            if chk and chk.isChecked():
                f_val = float(self.table.item(i, 1).text())
                self.on_save_callback(f_val, f"Pico de barrido ({self.table.item(i, 2).text()})")
                saved_count += 1

        QtWidgets.QMessageBox.information(
            self, "Registro actualizado", f"Se agregaron {saved_count} resonancias al registro de experimentos."
        )
        self.accept()


class RecordEditDialog(QtWidgets.QDialog):
    """Diálogo para ingresar o editar una fila del registro experimental con datos del afinador."""

    def __init__(self, freq_prog: float, note: str = "", mode_str: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guardar punto experimental de resonancia")
        self.resize(460, 300)

        self.freq_prog = freq_prog
        self.note = note
        self.mode_str = mode_str
        self.photo_path = ""
        self.record: Optional[ExperimentRecord] = None

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QFormLayout(self)

        self.prog_spin = QtWidgets.QDoubleSpinBox()
        self.prog_spin.setRange(10.0, 20000.0)
        self.prog_spin.setValue(self.freq_prog)
        self.prog_spin.setSuffix(" Hz")
        layout.addRow("Frecuencia generador:", self.prog_spin)

        self.tuner_spin = QtWidgets.QDoubleSpinBox()
        self.tuner_spin.setRange(0.0, 20000.0)
        self.tuner_spin.setValue(self.freq_prog)
        self.tuner_spin.setSpecialValueText("No medido")
        self.tuner_spin.setSuffix(" Hz")
        layout.addRow("Frecuencia afinador clip:", self.tuner_spin)

        self.note_edit = QtWidgets.QLineEdit(self.note)
        layout.addRow("Nota / Desviación:", self.note_edit)

        self.mode_edit = QtWidgets.QLineEdit(self.mode_str)
        layout.addRow("Modo observado:", self.mode_edit)

        photo_row = QtWidgets.QHBoxLayout()
        self.photo_label = QtWidgets.QLineEdit()
        self.photo_label.setPlaceholderText("Ruta opcional de foto de la figura...")
        photo_btn = QtWidgets.QPushButton("Examinar...")
        photo_btn.clicked.connect(self._browse_photo)
        photo_row.addWidget(self.photo_label)
        photo_row.addWidget(photo_btn)
        layout.addRow("Foto:", photo_row)

        btn_row = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("Guardar")
        ok_btn.clicked.connect(self._accept_record)
        cancel_btn = QtWidgets.QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row)

    def _browse_photo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Seleccionar foto de la figura de Chladni", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.photo_label.setText(path)

    def _accept_record(self):
        f_tuner = self.tuner_spin.value() if self.tuner_spin.value() > 0 else None
        self.record = ExperimentRecord(
            freq_prog=self.prog_spin.value(),
            freq_tuner=f_tuner,
            note=self.note_edit.text().strip(),
            mode_observed=self.mode_edit.text().strip(),
            photo_path=self.photo_label.text().strip()
        )
        self.accept()

