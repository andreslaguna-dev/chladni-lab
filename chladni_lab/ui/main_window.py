"""Ventana principal del Laboratorio de Chladni.

Integra generador de audio, monitor por cable, barrido, simulación de placa (cuadrada y circular),
asistente de calibración, registro con afinador y exportación CSV.
"""
from datetime import datetime
import math
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from chladni_lab.audio import InputMonitor, ToneGenerator, list_audio_devices
from chladni_lab.config import load_config, save_config
from chladni_lab.dsp import (FrequencySmoother, analyze_spectrum, calc_signal_metrics,
                            freq_to_note)
from chladni_lab.physics import (best_circular_mode, best_mode, circular_mode_table,
                                get_bessel_zero, mode_table)
from chladni_lab.simulation import SandSimulation
from chladni_lab.storage import ExperimentRecord, ExperimentStorage
from chladni_lab.ui.dialogs import (CalibrationDialog, RecordEditDialog,
                                   ResonanceSuggestionDialog)
from chladni_lab.ui.plots import (OscilloscopePlot, PlatePlot, SpectrumPlot,
                                 SweepResponsePlot)
from chladni_lab.ui.theme import apply_theme, get_stylesheet

FMIN = 20.0
FMAX = 5000.0


class MainWindow(QtWidgets.QMainWindow):
    """Ventana principal con interfaz de usuario en español y estilo Fusion."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laboratorio de figuras de Chladni — Acústica y Placas Vibrantes")
        self.resize(1300, 740)
        self.setMinimumSize(1024, 600)

        self.cfg = load_config()

        # Núcleo de audio y física
        self.tone = ToneGenerator(sr=self.cfg.get("sample_rate", 44100))
        self.monitor = InputMonitor(sr=self.cfg.get("sample_rate", 44100))
        self.sand = SandSimulation(
            count=self.cfg.get("grain_count", 7000),
            geometry=self.cfg.get("plate_type", "square")
        )
        self.storage = ExperimentStorage(Path("resonancias.json"))
        self.smoother = FrequencySmoother()

        # Estado del barrido
        self.sweep_timer = QtCore.QTimer(self)
        self.sweep_timer.timeout.connect(self._sweep_tick)
        self.sweep_running = False
        self.sweep_paused = False
        self.sweep_elapsed = 0.0
        self.sweep_direction = 1  # 1: subida, -1: bajada (ida y vuelta)
        self.sweep_freq_history: List[float] = []
        self.sweep_amp_history: List[float] = []
        self.detected_peaks: List[Tuple[float, float]] = []

        # Estado de la última captura
        self.last_dominant_freq: Optional[float] = None
        self.last_note_str: str = ""

        self._build_ui()
        self._load_state_to_ui()
        self._start_audio()

        # Temporizador de actualización visual (40 ms = 25 FPS)
        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self._update_view)
        self.ui_timer.start(40)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Panel izquierdo con barra de desplazamiento (ScrollArea para 1366x768)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(410)
        scroll_content = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(scroll_content)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # 1. Selector de Audio
        audio_group = QtWidgets.QGroupBox("1. Dispositivos de Audio")
        audio_form = QtWidgets.QFormLayout(audio_group)
        self.out_combo = QtWidgets.QComboBox()
        self.in_combo = QtWidgets.QComboBox()
        self._populate_audio_devices()
        self.out_combo.currentIndexChanged.connect(self._change_audio_device)
        self.in_combo.currentIndexChanged.connect(self._change_audio_device)
        audio_form.addRow("Salida:", self.out_combo)
        audio_form.addRow("Entrada:", self.in_combo)
        left_layout.addWidget(audio_group)

        # 2. Generador de Audio
        gen_group = QtWidgets.QGroupBox("2. Generador de Señal")
        gen_layout = QtWidgets.QVBoxLayout(gen_group)

        freq_header = QtWidgets.QHBoxLayout()
        freq_header.addWidget(QtWidgets.QLabel("<b>Frecuencia:</b>"))
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setRange(FMIN, FMAX)
        self.freq_spin.setDecimals(2)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.setValue(self.cfg.get("frequency", 200.0))
        self.freq_spin.valueChanged.connect(self._on_freq_spin_changed)
        freq_header.addWidget(self.freq_spin)
        gen_layout.addLayout(freq_header)

        # Micro-ajustes de frecuencia (±0.1 y ±0.01 Hz)
        fine_row = QtWidgets.QHBoxLayout()
        btn_m01 = QtWidgets.QPushButton("-0.1")
        btn_p01 = QtWidgets.QPushButton("+0.1")
        btn_m001 = QtWidgets.QPushButton("-0.01")
        btn_p001 = QtWidgets.QPushButton("+0.01")
        btn_m01.clicked.connect(lambda: self._adjust_freq(-0.1))
        btn_p01.clicked.connect(lambda: self._adjust_freq(0.1))
        btn_m001.clicked.connect(lambda: self._adjust_freq(-0.01))
        btn_p001.clicked.connect(lambda: self._adjust_freq(0.01))
        fine_row.addWidget(btn_m01)
        fine_row.addWidget(btn_p01)
        fine_row.addWidget(btn_m001)
        fine_row.addWidget(btn_p001)
        gen_layout.addLayout(fine_row)

        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.freq_slider.setRange(int(FMIN * 10), int(FMAX * 10))
        self.freq_slider.setValue(int(self.freq_spin.value() * 10))
        self.freq_slider.valueChanged.connect(self._on_freq_slider_changed)
        gen_layout.addWidget(self.freq_slider)

        # Volumen y limitador
        vol_row = QtWidgets.QHBoxLayout()
        vol_row.addWidget(QtWidgets.QLabel("Volumen:"))
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(self.cfg.get("volume", 0.15) * 100))
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        self.vol_label = QtWidgets.QLabel(f"{self.vol_slider.value()}%")
        vol_row.addWidget(self.vol_slider)
        vol_row.addWidget(self.vol_label)
        gen_layout.addLayout(vol_row)

        self.vol_warn_label = QtWidgets.QLabel("⚠️ ¡Precaución! Volumen alto (> 70%)")
        self.vol_warn_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.vol_warn_label.setVisible(self.vol_slider.value() > 70)
        gen_layout.addWidget(self.vol_warn_label)

        self.play_btn = QtWidgets.QPushButton("▶ Reproducir tono")
        self.play_btn.setCheckable(True)
        self.play_btn.setFixedHeight(34)
        self.play_btn.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.play_btn.toggled.connect(self._toggle_play)
        gen_layout.addWidget(self.play_btn)
        left_layout.addWidget(gen_group)

        # 3. Barrido de Frecuencia
        sweep_group = QtWidgets.QGroupBox("3. Barrido de Frecuencia (Búsqueda de Resonancias)")
        sw_layout = QtWidgets.QGridLayout(sweep_group)

        self.sw_from = QtWidgets.QDoubleSpinBox()
        self.sw_to = QtWidgets.QDoubleSpinBox()
        for sp, v in ((self.sw_from, 100.0), (self.sw_to, 1200.0)):
            sp.setRange(FMIN, FMAX)
            sp.setValue(v)
            sp.setSuffix(" Hz")
        sw_layout.addWidget(QtWidgets.QLabel("Desde:"), 0, 0)
        sw_layout.addWidget(self.sw_from, 0, 1)
        sw_layout.addWidget(QtWidgets.QLabel("Hasta:"), 0, 2)
        sw_layout.addWidget(self.sw_to, 0, 3)

        self.sw_time = QtWidgets.QDoubleSpinBox()
        self.sw_time.setRange(5, 600)
        self.sw_time.setValue(45)
        self.sw_time.setSuffix(" s")
        sw_layout.addWidget(QtWidgets.QLabel("Duración:"), 1, 0)
        sw_layout.addWidget(self.sw_time, 1, 1)

        self.sw_log_check = QtWidgets.QCheckBox("Logarítmico")
        self.sw_log_check.setChecked(True)
        self.sw_pingpong_check = QtWidgets.QCheckBox("Ida y vuelta")
        sw_layout.addWidget(self.sw_log_check, 1, 2)
        sw_layout.addWidget(self.sw_pingpong_check, 1, 3)

        sw_btn_row = QtWidgets.QHBoxLayout()
        self.sw_start_btn = QtWidgets.QPushButton("Iniciar barrido")
        self.sw_start_btn.setCheckable(True)
        self.sw_start_btn.toggled.connect(self._toggle_sweep)
        self.sw_pause_btn = QtWidgets.QPushButton("Pausa")
        self.sw_pause_btn.setEnabled(False)
        self.sw_pause_btn.clicked.connect(self._toggle_sweep_pause)
        sw_btn_row.addWidget(self.sw_start_btn)
        sw_btn_row.addWidget(self.sw_pause_btn)
        sw_layout.addLayout(sw_btn_row, 2, 0, 1, 4)

        self.sw_suggest_btn = QtWidgets.QPushButton("🔎 Analizar picos de resonancia")
        self.sw_suggest_btn.clicked.connect(self._show_resonance_suggestions)
        sw_layout.addWidget(self.sw_suggest_btn, 3, 0, 1, 4)
        left_layout.addWidget(sweep_group)

        # 4. Simulación y Física de la Placa
        sim_group = QtWidgets.QGroupBox("4. Parámetros de la Placa")
        sim_layout = QtWidgets.QGridLayout(sim_group)

        sim_layout.addWidget(QtWidgets.QLabel("Geometría:"), 0, 0)
        self.geom_combo = QtWidgets.QComboBox()
        self.geom_combo.addItems(["Placa Cuadrada", "Placa Circular"])
        self.geom_combo.currentIndexChanged.connect(self._change_geometry)
        sim_layout.addWidget(self.geom_combo, 0, 1, 1, 2)

        sim_layout.addWidget(QtWidgets.QLabel("f0 base:"), 1, 0)
        self.f0_spin = QtWidgets.QDoubleSpinBox()
        self.f0_spin.setRange(1.0, 5000.0)
        self.f0_spin.setValue(self.cfg.get("f0", 100.0))
        self.f0_spin.setSuffix(" Hz")
        sim_layout.addWidget(self.f0_spin, 1, 1)

        self.exp_spin = QtWidgets.QDoubleSpinBox()
        self.exp_spin.setRange(0.2, 3.0)
        self.exp_spin.setValue(self.cfg.get("exponent", 1.0))
        self.exp_spin.setDecimals(3)
        self.exp_spin.setSingleStep(0.05)
        sim_layout.addWidget(QtWidgets.QLabel("Exponente p:"), 2, 0)
        sim_layout.addWidget(self.exp_spin, 2, 1)

        calib_btn = QtWidgets.QPushButton("⚖ Calibrar (Mínimos Cuadrados)")
        calib_btn.clicked.connect(self._open_calibration_wizard)
        sim_layout.addWidget(calib_btn, 1, 2, 2, 1)

        self.force_check = QtWidgets.QCheckBox("Forzar modo manual:")
        self.force_check.toggled.connect(self._toggle_force_mode)
        sim_layout.addWidget(self.force_check, 3, 0, 1, 3)

        mode_num_row = QtWidgets.QHBoxLayout()
        self.lbl_n = QtWidgets.QLabel("n:")
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(0, 12)
        self.spin_n.setValue(1)
        self.lbl_m = QtWidgets.QLabel("m:")
        self.spin_m = QtWidgets.QSpinBox()
        self.spin_m.setRange(1, 12)
        self.spin_m.setValue(2)
        self.sign_combo = QtWidgets.QComboBox()
        self.sign_combo.addItems(["Simétrico (+)", "Antisimétrico (-)"])
        mode_num_row.addWidget(self.lbl_n)
        mode_num_row.addWidget(self.spin_n)
        mode_num_row.addWidget(self.lbl_m)
        mode_num_row.addWidget(self.spin_m)
        mode_num_row.addWidget(self.sign_combo)
        sim_layout.addLayout(mode_num_row, 4, 0, 1, 3)

        # Controles de partículas y visualización
        self.heat_check = QtWidgets.QCheckBox("Mostrar líneas nodales teóricas")
        self.heat_check.setChecked(True)
        sim_layout.addWidget(self.heat_check, 5, 0, 1, 2)

        reset_sand_btn = QtWidgets.QPushButton("↻ Sacudir arena")
        reset_sand_btn.clicked.connect(self.sand.reset)
        sim_layout.addWidget(reset_sand_btn, 5, 2)

        # Ajuste de granos y velocidad
        sand_cfg_row = QtWidgets.QHBoxLayout()
        sand_cfg_row.addWidget(QtWidgets.QLabel("Granos:"))
        self.grain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.grain_slider.setRange(2000, 12000)
        self.grain_slider.setValue(self.cfg.get("grain_count", 7000))
        self.grain_slider.valueChanged.connect(self._on_grain_count_changed)
        self.grain_label = QtWidgets.QLabel(str(self.grain_slider.value()))
        sand_cfg_row.addWidget(self.grain_slider)
        sand_cfg_row.addWidget(self.grain_label)
        sim_layout.addLayout(sand_cfg_row, 6, 0, 1, 3)

        snap_btn = QtWidgets.QPushButton("📷 Guardar captura de la figura (PNG)")
        snap_btn.clicked.connect(self._save_plate_screenshot)
        sim_layout.addWidget(snap_btn, 7, 0, 1, 3)

        left_layout.addWidget(sim_group)

        # Barra inferior de tema
        bottom_row = QtWidgets.QHBoxLayout()
        self.theme_btn = QtWidgets.QPushButton("🌓 Cambiar Tema")
        self.theme_btn.clicked.connect(self._toggle_theme)
        bottom_row.addWidget(self.theme_btn)
        left_layout.addLayout(bottom_row)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 0)

        # ---- Panel derecho (Visualizaciones y Datos)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Encabezado de estado de la señal
        status_card = QtWidgets.QFrame()
        status_card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        status_layout = QtWidgets.QHBoxLayout(status_card)
        status_layout.setContentsMargins(12, 6, 12, 6)

        self.freq_readout = QtWidgets.QLabel("Entrada: —")
        self.freq_readout.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a90e2;")
        self.note_readout = QtWidgets.QLabel("")
        self.note_readout.setStyleSheet("font-size: 14px; color: #9aa0a6;")

        self.meter_bar = QtWidgets.QProgressBar()
        self.meter_bar.setRange(0, 100)
        self.meter_bar.setValue(0)
        self.meter_bar.setTextVisible(False)
        self.meter_bar.setFixedWidth(130)
        self.meter_bar.setFixedHeight(14)

        self.clip_badge = QtWidgets.QLabel(" CLIP ")
        self.clip_badge.setStyleSheet("background-color: #aa2222; color: white; font-weight: bold; border-radius: 3px;")
        self.clip_badge.setVisible(False)

        status_layout.addWidget(self.freq_readout)
        status_layout.addWidget(self.note_readout)
        status_layout.addStretch()
        status_layout.addWidget(QtWidgets.QLabel("Nivel:"))
        status_layout.addWidget(self.meter_bar)
        status_layout.addWidget(self.clip_badge)
        right_layout.addWidget(status_card)

        # Pestañas principales
        self.tabs = QtWidgets.QTabWidget()

        # Tab 1: Simulación de Placa y Espectro en vivo
        tab_live = QtWidgets.QWidget()
        live_layout = QtWidgets.QHBoxLayout(tab_live)
        live_layout.setContentsMargins(4, 4, 4, 4)

        self.plate_plot = PlatePlot()
        live_layout.addWidget(self.plate_plot, 3)

        plots_col = QtWidgets.QVBoxLayout()
        self.spec_plot = SpectrumPlot()
        self.osc_plot = OscilloscopePlot()
        plots_col.addWidget(self.spec_plot, 1)
        plots_col.addWidget(self.osc_plot, 1)
        live_layout.addLayout(plots_col, 2)

        self.tabs.addTab(tab_live, "🎮 Simulación de Placa y Audio en Vivo")

        # Tab 2: Curva de Barrido
        tab_sweep = QtWidgets.QWidget()
        sweep_tab_layout = QtWidgets.QVBoxLayout(tab_sweep)
        self.sweep_plot = SweepResponsePlot()
        sweep_tab_layout.addWidget(self.sweep_plot)

        sweep_disclaimer = QtWidgets.QLabel(
            "<i>Nota: Con el cable 1 a 1 de retorno, la gráfica registra la respuesta de frecuencia eléctrica "
            "del convertidor D/A a A/D. Para medir la curva mecánica de la placa, conecte un micrófono o sensor piezoeléctrico.</i>"
        )
        sweep_disclaimer.setStyleSheet("color: #888; font-size: 11px;")
        sweep_tab_layout.addWidget(sweep_disclaimer)
        self.tabs.addTab(tab_sweep, "📈 Curva de Respuesta de Barrido A(f)")

        # Tab 3: Registro de Experimentos
        tab_exp = QtWidgets.QWidget()
        exp_layout = QtWidgets.QVBoxLayout(tab_exp)

        exp_btn_row = QtWidgets.QHBoxLayout()
        add_exp_btn = QtWidgets.QPushButton("+ Guardar punto actual")
        add_exp_btn.clicked.connect(self._add_current_experiment)
        csv_btn = QtWidgets.QPushButton("📁 Exportar a CSV")
        csv_btn.clicked.connect(self._export_experiments_csv)
        del_exp_btn = QtWidgets.QPushButton("🗑 Eliminar seleccionado")
        del_exp_btn.clicked.connect(self._delete_selected_experiment)
        exp_btn_row.addWidget(add_exp_btn)
        exp_btn_row.addWidget(csv_btn)
        exp_btn_row.addWidget(del_exp_btn)
        exp_btn_row.addStretch()
        exp_layout.addLayout(exp_btn_row)

        self.exp_table = QtWidgets.QTableWidget(0, 7)
        self.exp_table.setHorizontalHeaderLabels([
            "f Generador (Hz)",
            "f Afinador (Hz)",
            "Nota / Centésimas",
            "Error Relativo (%)",
            "Modo Observado",
            "Foto",
            "Fecha"
        ])
        self.exp_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.exp_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.exp_table.itemDoubleClicked.connect(self._recall_experiment)
        self.exp_table.itemChanged.connect(self._on_table_item_edited)
        exp_layout.addWidget(self.exp_table)
        self.tabs.addTab(tab_exp, "📋 Registro de Experimentos y Afinador")

        right_layout.addWidget(self.tabs, 1)

        # Pie de ventana: descripción del modo actual
        self.mode_status_label = QtWidgets.QLabel("")
        self.mode_status_label.setStyleSheet("padding: 4px; font-size: 12px; color: #a0a6b5;")
        right_layout.addWidget(self.mode_status_label)

        main_layout.addWidget(right_panel, 1)
        self._refresh_experiment_table()

    def _load_state_to_ui(self):
        geom = self.cfg.get("plate_type", "square")
        idx = 1 if geom == "circular" else 0
        self.geom_combo.setCurrentIndex(idx)
        self.plate_plot.set_geometry(geom)
        self.sand.set_geometry(geom)
        sign_saved = self.cfg.get("sign", 1)
        self.sign_combo.setCurrentIndex(0 if sign_saved >= 0 else 1)
        self._toggle_force_mode(self.force_check.isChecked())

    # ------------------------------------------------------------- Dispositivos
    def _populate_audio_devices(self):
        self.out_combo.blockSignals(True)
        self.in_combo.blockSignals(True)
        self.out_combo.clear()
        self.in_combo.clear()

        out_devs = list_audio_devices("output")
        in_devs = list_audio_devices("input")

        for d in out_devs:
            self.out_combo.addItem(d["label"], d["index"])
        for d in in_devs:
            self.in_combo.addItem(d["label"], d["index"])

        # Seleccionar dispositivo guardado o primer recomendado
        saved_out = self.cfg.get("output_device")
        saved_in = self.cfg.get("input_device")

        self._select_device_in_combo(self.out_combo, saved_out)
        self._select_device_in_combo(self.in_combo, saved_in)

        self.out_combo.blockSignals(False)
        self.in_combo.blockSignals(False)

    @staticmethod
    def _select_device_in_combo(combo: QtWidgets.QComboBox, dev_idx: Optional[int]):
        if dev_idx is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == dev_idx:
                    combo.setCurrentIndex(i)
                    return
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    def _start_audio(self):
        out_idx = self.out_combo.currentData()
        in_idx = self.in_combo.currentData()
        ok_out, msg_out = self.tone.start(device=out_idx)
        ok_in, msg_in = self.monitor.start(device=in_idx)
        if not ok_out or not ok_in:
            self.mode_status_label.setText(f"Aviso de audio: Salida: {msg_out} | Entrada: {msg_in}")

    def _change_audio_device(self):
        out_idx = self.out_combo.currentData()
        in_idx = self.in_combo.currentData()
        self.cfg["output_device"] = out_idx
        self.cfg["input_device"] = in_idx
        save_config(self.cfg)
        self._start_audio()

    # ---------------------------------------------------------------- Generador
    def _on_freq_spin_changed(self, val: float):
        self.tone.frequency = val
        self.freq_slider.blockSignals(True)
        self.freq_slider.setValue(int(round(val * 10)))
        self.freq_slider.blockSignals(False)

    def _on_freq_slider_changed(self, val: int):
        self.freq_spin.setValue(val / 10.0)

    def _adjust_freq(self, delta: float):
        curr = self.freq_spin.value()
        self.freq_spin.setValue(max(FMIN, min(FMAX, round(curr + delta, 3))))

    def _on_volume_changed(self, val: int):
        vol_fraction = val / 100.0
        self.tone.volume = vol_fraction
        self.vol_label.setText(f"{val}%")
        self.vol_warn_label.setVisible(val > 70)
        self.cfg["volume"] = vol_fraction
        save_config(self.cfg)

    def _toggle_play(self, playing: bool):
        self.tone.playing = playing
        if playing:
            self.play_btn.setText("■ Detener tono")
        else:
            self.play_btn.setText("▶ Reproducir tono")

    # ------------------------------------------------------------------ Barrido
    def _toggle_sweep(self, start: bool):
        if start:
            self.sweep_running = True
            self.sweep_paused = False
            self.sweep_freq_history.clear()
            self.sweep_amp_history.clear()
            self.sweep_plot.clear_sweep()
            self.sw_pause_btn.setEnabled(True)
            self.sw_pause_btn.setText("Pausar")
            self.sw_start_btn.setText("Detener barrido")
            self.play_btn.setChecked(True)
            self.tabs.setCurrentIndex(1)  # Mostrar pestaña de curva de barrido

            # Iniciar barrido muestra a muestra directamente en el hilo de audio
            f0 = self.sw_from.value()
            f1 = self.sw_to.value()
            duration = max(5.0, self.sw_time.value())
            is_log = self.sw_log_check.isChecked()
            is_pingpong = self.sw_pingpong_check.isChecked()
            self.tone.start_sweep(f0, f1, duration, log=is_log, pingpong=is_pingpong)

            # Temporizador de GUI a 50 ms para refrescar curva en pantalla
            self.sweep_timer.start(50)
        else:
            self.sweep_running = False
            self.sweep_paused = False
            self.sweep_timer.stop()
            self.tone.stop_sweep()
            self.sw_start_btn.setText("Iniciar barrido")
            self.sw_pause_btn.setEnabled(False)
            self._analyze_sweep_peaks()

    def _toggle_sweep_pause(self):
        if not self.sweep_running:
            return
        self.sweep_paused = not self.sweep_paused
        if self.sweep_paused:
            self.tone.pause_sweep()
            self.sweep_timer.stop()
            self.sw_pause_btn.setText("Reanudar")
        else:
            self.tone.resume_sweep()
            self.sweep_timer.start(50)
            self.sw_pause_btn.setText("Pausar")

    def _sweep_tick(self):
        if self.sweep_paused:
            return

        if not self.tone.sweep_active:
            self.sw_start_btn.setChecked(False)
            return

        current_f = self.tone.current_frequency

        # Reflejar la frecuencia actual en los controles sin alterar el generador
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(current_f)
        self.freq_slider.blockSignals(True)
        self.freq_slider.setValue(int(round(current_f * 10)))
        self.freq_slider.blockSignals(False)
        self.freq_spin.blockSignals(False)

        # Registrar amplitud de entrada por el cable 3.5 mm
        samples = self.monitor.snapshot(2048)
        rms, _, _ = calc_signal_metrics(samples)
        self.sweep_freq_history.append(current_f)
        self.sweep_amp_history.append(rms)
        self.sweep_plot.update_data(self.sweep_freq_history, self.sweep_amp_history, current_f)

    def _analyze_sweep_peaks(self):
        if len(self.sweep_amp_history) < 10:
            return

        amps = np.array(self.sweep_amp_history)
        freqs = np.array(self.sweep_freq_history)

        # Ordenar por frecuencia
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        amps = amps[sort_idx]

        # Búsqueda de máximos locales significativos
        peaks = []
        threshold = np.mean(amps) + 0.5 * np.std(amps)
        for i in range(1, len(amps) - 1):
            if amps[i] > amps[i - 1] and amps[i] > amps[i + 1] and amps[i] > threshold:
                peaks.append((float(freqs[i]), float(amps[i])))

        # Filtrar picos demasiado juntos (< 5 Hz)
        filtered_peaks = []
        for f, a in peaks:
            if not filtered_peaks or abs(f - filtered_peaks[-1][0]) > 5.0:
                filtered_peaks.append((f, a))

        self.detected_peaks = filtered_peaks
        if filtered_peaks:
            pf = [p[0] for p in filtered_peaks]
            pa = [p[1] for p in filtered_peaks]
            self.sweep_plot.set_detected_peaks(pf, pa)

    def _show_resonance_suggestions(self):
        if not self.detected_peaks:
            self._analyze_sweep_peaks()

        if not self.detected_peaks:
            QtWidgets.QMessageBox.information(
                self, "Barrido", "No se encontraron picos de resonancia destacados. Ejecuta un barrido primero."
            )
            return

        def save_peak(f_val, note_text):
            rec = ExperimentRecord(freq_prog=f_val, note=note_text, mode_observed="Candidato de barrido")
            self.storage.add(rec)
            self._refresh_experiment_table()

        dlg = ResonanceSuggestionDialog(self.detected_peaks, on_save_callback=save_peak, parent=self)
        dlg.exec_()

    # ------------------------------------------------------------- Física y Sim
    def _change_geometry(self, idx: int):
        geom = "circular" if idx == 1 else "square"
        self.plate_plot.set_geometry(geom)
        self.sand.set_geometry(geom)
        if geom == "circular":
            self.lbl_n.setText("m (diám):")
            self.lbl_m.setText("n (circ):")
            self.sign_combo.setVisible(False)
        else:
            self.lbl_n.setText("n:")
            self.lbl_m.setText("m:")
            self.sign_combo.setVisible(True)
        self.cfg["plate_type"] = geom
        save_config(self.cfg)

    def _toggle_force_mode(self, checked: bool):
        self.spin_n.setEnabled(checked)
        self.spin_m.setEnabled(checked)
        self.sign_combo.setEnabled(checked)

    def _on_grain_count_changed(self, val: int):
        self.grain_label.setText(str(val))
        self.sand.set_count(val)
        self.cfg["grain_count"] = val
        save_config(self.cfg)

    def _open_calibration_wizard(self):
        geom = "circular" if self.geom_combo.currentIndex() == 1 else "square"
        dlg = CalibrationDialog(self.f0_spin.value(), self.exp_spin.value(), geometry=geom, parent=self)
        if dlg.exec_() and dlg.applied:
            self.f0_spin.setValue(dlg.calibrated_f0)
            self.exp_spin.setValue(dlg.calibrated_p)
            self.cfg["f0"] = dlg.calibrated_f0
            self.cfg["exponent"] = dlg.calibrated_p
            save_config(self.cfg)

    def _save_plate_screenshot(self):
        geom = "circular" if self.geom_combo.currentIndex() == 1 else "cuadrada"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"chladni_{geom}_{int(self.tone.frequency)}Hz_{ts}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar figura de Chladni", default_name, "Imágenes PNG (*.png)"
        )
        if path:
            ok = self.plate_plot.grab_screenshot(Path(path))
            if ok:
                QtWidgets.QMessageBox.information(self, "Captura", f"Figura guardada con éxito en:\n{path}")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "No se pudo guardar la captura de imagen.")

    # ------------------------------------------------------------- Experimentos
    def _add_current_experiment(self):
        f_val = self.tone.frequency
        mode_str = self.mode_status_label.text().split("·")[0].strip()
        dlg = RecordEditDialog(freq_prog=f_val, note=self.last_note_str, mode_str=mode_str, parent=self)
        if dlg.exec_() and dlg.record:
            self.storage.add(dlg.record)
            self._refresh_experiment_table()

    def _refresh_experiment_table(self):
        self.exp_table.blockSignals(True)
        self.exp_table.setRowCount(len(self.storage.records))
        for r, rec in enumerate(self.storage.records):
            item_f = QtWidgets.QTableWidgetItem(f"{rec.freq_prog:.2f}")
            item_f.setFlags(item_f.flags() & ~QtCore.Qt.ItemIsEditable)

            item_tuner = QtWidgets.QTableWidgetItem(f"{rec.freq_tuner:.2f}" if rec.freq_tuner else "")
            item_note = QtWidgets.QTableWidgetItem(rec.note)
            item_err = QtWidgets.QTableWidgetItem(f"{rec.error_rel:.2f}%" if rec.error_rel is not None else "—")
            item_err.setFlags(item_err.flags() & ~QtCore.Qt.ItemIsEditable)

            item_mode = QtWidgets.QTableWidgetItem(rec.mode_observed)
            item_photo = QtWidgets.QTableWidgetItem(rec.photo_path)
            item_date = QtWidgets.QTableWidgetItem(rec.timestamp)
            item_date.setFlags(item_date.flags() & ~QtCore.Qt.ItemIsEditable)

            self.exp_table.setItem(r, 0, item_f)
            self.exp_table.setItem(r, 1, item_tuner)
            self.exp_table.setItem(r, 2, item_note)
            self.exp_table.setItem(r, 3, item_err)
            self.exp_table.setItem(r, 4, item_mode)
            self.exp_table.setItem(r, 5, item_photo)
            self.exp_table.setItem(r, 6, item_date)
        self.exp_table.blockSignals(False)

    def _on_table_item_edited(self, item: QtWidgets.QTableWidgetItem):
        row = item.row()
        col = item.column()
        if 0 <= row < len(self.storage.records):
            rec = self.storage.records[row]
            if col == 1:
                # Editó frecuencia de afinador
                txt = item.text().strip()
                try:
                    rec.freq_tuner = float(txt) if txt else None
                except ValueError:
                    rec.freq_tuner = None
            elif col == 2:
                rec.note = item.text().strip()
            elif col == 4:
                rec.mode_observed = item.text().strip()
            elif col == 5:
                rec.photo_path = item.text().strip()
            self.storage.save()
            self._refresh_experiment_table()

    def _recall_experiment(self, item: QtWidgets.QTableWidgetItem):
        row = item.row()
        if 0 <= row < len(self.storage.records):
            target_f = self.storage.records[row].freq_prog
            self.freq_spin.setValue(target_f)

    def _delete_selected_experiment(self):
        row = self.exp_table.currentRow()
        if row >= 0:
            self.storage.remove(row)
            self._refresh_experiment_table()

    def _export_experiments_csv(self):
        default_name = f"experimentos_chladni_{datetime.now().strftime('%Y%m%d')}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar registros a CSV", default_name, "Archivos CSV (*.csv)"
        )
        if path:
            ok = self.storage.export_csv(Path(path))
            if ok:
                QtWidgets.QMessageBox.information(self, "Exportar CSV", f"Registros exportados exitosamente en:\n{path}")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "No se pudo exportar el archivo CSV.")

    # ----------------------------------------------------------- Tema y Estilos
    def _toggle_theme(self):
        new_theme = "light" if self.cfg.get("theme", "dark") == "dark" else "dark"
        self.cfg["theme"] = new_theme
        save_config(self.cfg)
        app = QtWidgets.QApplication.instance()
        if app:
            apply_theme(app, new_theme)
            app.setStyleSheet(get_stylesheet(new_theme))

    # ------------------------------------------------------------- Loop Visual
    def _update_view(self):
        # 1. Procesar entrada de audio (cable 3.5 mm)
        samples = self.monitor.snapshot(4096)
        sr = self.monitor.sr or 44100
        rms, peak_val, is_clip = calc_signal_metrics(samples)

        # Actualizar medidor de nivel VU y saturación
        self.meter_bar.setValue(int(min(1.0, rms * 4.0) * 100))
        self.clip_badge.setVisible(is_clip)

        # Análisis espectral
        freqs, spec, raw_peak = analyze_spectrum(samples, sr=sr)
        smoothed_peak = self.smoother.update(raw_peak)
        self.last_dominant_freq = smoothed_peak

        if self.tabs.currentIndex() == 0:
            self.spec_plot.update_spectrum(freqs, spec, smoothed_peak)
            self.osc_plot.update_waveform(samples, sr=sr)

        if smoothed_peak is not None:
            note_str, cents = freq_to_note(smoothed_peak)
            self.last_note_str = f"{note_str} ({cents:+.0f}¢)"
            self.freq_readout.setText(f"Entrada: {smoothed_peak:.2f} Hz")
            self.note_readout.setText(f"Nota más cercana: {note_str} ({cents:+.0f} centésimas)")
        else:
            self.freq_readout.setText("Entrada: sin señal")
            self.note_readout.setText("Verifica la conexión del cable de audio o sube el volumen")
            self.last_note_str = ""

        # 2. Simulación de la placa
        is_circular = self.geom_combo.currentIndex() == 1
        f0 = self.f0_spin.value()
        p = self.exp_spin.value()
        gen_freq = self.tone.frequency
        playing = self.tone.playing

        sign = 1 if self.sign_combo.currentIndex() == 0 else -1

        if self.force_check.isChecked():
            n = self.spin_n.value()
            m = self.spin_m.value()
            if is_circular:
                base_alpha = get_bessel_zero(0, 1)
                alpha = get_bessel_zero(n, max(1, m))
                fm = f0 * (((alpha / base_alpha) ** 2) ** p)
            else:
                fm = f0 * ((n * n + m * m) ** p)
            strength = 1.0 if playing else 0.0
        else:
            if is_circular:
                n, m, fm, strength_val = best_circular_mode(gen_freq, f0, exponent=p)
            else:
                n, m, fm, strength_val = best_mode(gen_freq, f0, exponent=p)
            strength = strength_val if playing else 0.0

        degenerate_null = (not is_circular) and (n == m) and (sign == -1)
        if degenerate_null:
            strength = 0.0

        # Actualizar líneas nodales y arena (solo cuando la pestaña de la placa está activa)
        if self.tabs.currentIndex() == 0:
            self.plate_plot.update_nodal_lines(n, m, sign=sign, visible=self.heat_check.isChecked())
            xs, ys = self.sand.step(n, m, strength, jitter=0.04, sign=sign)
            self.plate_plot.update_sand(xs, ys)

        # Texto informativo inferior
        geom_name = "Circular" if is_circular else "Cuadrada"
        mode_label = f"m={n}, n={m}" if is_circular else f"n={n}, m={m}"
        sym_label = "" if is_circular else (" (simétrico)" if sign == 1 else " (antisimétrico)")
        if degenerate_null:
            self.mode_status_label.setText(
                f"Placa {geom_name}: Modo {mode_label}{sym_label} · "
                "Modo nulo por simetría (con n = m elige Simétrico (+) o cambia n ≠ m)"
            )
        else:
            self.mode_status_label.setText(
                f"Placa {geom_name}: Modo {mode_label}{sym_label} · Frecuencia teórica: {fm:.1f} Hz · "
                f"Resonancia simulada: {strength * 100:.0f}%"
            )

    def closeEvent(self, event):
        self.cfg["frequency"] = self.freq_spin.value()
        self.cfg["volume"] = self.vol_slider.value() / 100.0
        self.cfg["f0"] = self.f0_spin.value()
        self.cfg["exponent"] = self.exp_spin.value()
        self.cfg["plate_type"] = "circular" if self.geom_combo.currentIndex() == 1 else "square"
        self.cfg["grain_count"] = self.grain_slider.value()
        self.cfg["sign"] = 1 if self.sign_combo.currentIndex() == 0 else -1
        self.cfg["output_device"] = self.out_combo.currentData()
        self.cfg["input_device"] = self.in_combo.currentData()
        save_config(self.cfg)
        self.tone.stop()
        self.monitor.stop()
        super().closeEvent(event)
