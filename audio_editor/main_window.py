"""Main application window for ai-audio-toolkit."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QGuiApplication, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QProgressBar, QGroupBox,
    QSplitter, QMessageBox, QStatusBar, QScrollArea,
    QTreeWidget, QTreeWidgetItem, QSizePolicy, QRadioButton,
    QButtonGroup, QLineEdit, QSlider, QCheckBox, QDialog, QTabWidget,
)

from .separator_backend import (
    MODEL_PRESETS, ModelPreset, get_preset_by_name,
    SeparationWorker, BatchSeparationWorker,
)
from .waveform_widget import WaveformWidget
from .audio_engine import AudioEngine
from .stem_track_widget import StemTrackWidget
from .transport_bar import TransportBar
from .log_viewer import LogViewerWidget
from .analysis import AudioAnalysisWorker
from .effects import (
    EFFECT_PRESETS,
    apply_preset,
    apply_eq,
    apply_compressor,
    EQBand,
    apply_effect_stack,
    empty_effect_state,
    normalize_effect_state,
)
from .effects_panel import LayerEffectsDialog
from .effects_preview import LiveEffectsPreviewProcessor
from .sam_backend import check_sam_available
from .enhance_backend import (
    check_enhance_available, check_deepfilter_available,
    MDXNET_ENHANCEMENT_PRESETS,
    check_audio_separator_available,
    check_clearvoice_available,
    check_voicefixer_available,
    check_metricgan_available,
    get_backend_probe_state,
)
from .diagnostics import probe_acceleration
from .media_utils import extract_audio_from_video, extract_audio_from_video_to_path
from .runtime import get_icon_path
from .branding import APP_DISPLAY_NAME
from .settings import get_app_settings
from .settings_dialog import SettingsDialog
from .split_dialog import SplitClipDialog


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', sans-serif;
}
QGroupBox {
    border: 1px solid #333344;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #cdd6f4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
}
QPushButton#primaryBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}
QPushButton#primaryBtn:hover { background-color: #b4d0fb; }
QPushButton#primaryBtn:disabled {
    background-color: #45475a;
    color: #585b70;
}
QPushButton#dangerBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px 10px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
}
QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    height: 22px;
}
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QTreeWidget {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #333344;
    border-radius: 4px;
    outline: none;
}
QTreeWidget::item:selected { background-color: #45475a; }
QTreeWidget::item:hover { background-color: #313244; }
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #333344;
}
QMenuBar { background-color: #181825; color: #cdd6f4; }
QMenuBar::item:selected { background-color: #313244; }
QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}
QMenu::item:selected { background-color: #45475a; }
QRadioButton { color: #cdd6f4; spacing: 6px; }
QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 2px solid #585b70;
    border-radius: 7px;
    background-color: #313244;
}
QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QRadioButton::indicator:hover {
    border-color: #89b4fa;
}
QTabWidget::pane {
    border: 1px solid #333344;
    border-radius: 0 0 4px 4px;
    background-color: #1e1e2e;
}
QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    border: 1px solid #333344;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 6px 18px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background-color: #313244;
    color: #cdd6f4;
    font-weight: bold;
    border-color: #45475a;
}
QTabBar::tab:hover:!selected { background-color: #252535; }
QSplitter::handle { background-color: #333344; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: #181825; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background: #45475a; min-height: 30px; border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px 10px;
}
QCheckBox { color: #cdd6f4; spacing: 6px; }
"""


class EffectsDialog(QDialog):
    """Dialog for per-stem or global effects with preview."""

    def __init__(self, audio_data: np.ndarray, sample_rate: int,
                 engine: AudioEngine, title: str = "Effects", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setStyleSheet(DARK_STYLE)
        self._audio = audio_data
        self._sr = sample_rate
        self._engine = engine
        self._result: np.ndarray | None = None

        layout = QVBoxLayout(self)

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["None"] + list(EFFECT_PRESETS.keys()))
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, stretch=1)
        layout.addLayout(preset_row)

        # EQ sliders
        eq_group = QGroupBox("Equalizer")
        eq_layout = QVBoxLayout(eq_group)
        self._eq_sliders = {}
        for name, freq in [("Bass (80 Hz)", 80), ("Mid (2.5 kHz)", 2500), ("Treble (10 kHz)", 10000)]:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(120)
            row.addWidget(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-120, 120)
            slider.setValue(0)
            row.addWidget(slider, stretch=1)
            val_lbl = QLabel("0 dB")
            val_lbl.setFixedWidth(50)
            slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(f"{v/10:.1f} dB"))
            row.addWidget(val_lbl)
            eq_layout.addLayout(row)
            self._eq_sliders[freq] = slider
        layout.addWidget(eq_group)

        # Compressor
        comp_group = QGroupBox("Compressor")
        comp_layout = QVBoxLayout(comp_group)
        self._comp_sliders = {}
        for name, key, lo, hi, default in [
            ("Threshold", "threshold_db", -40, 0, -20),
            ("Ratio", "ratio", 10, 200, 40),
            ("Attack (ms)", "attack_ms", 1, 100, 5),
            ("Release (ms)", "release_ms", 10, 500, 50),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(120)
            row.addWidget(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            row.addWidget(slider, stretch=1)
            val_lbl = QLabel(str(default))
            val_lbl.setFixedWidth(50)
            if key == "ratio":
                slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(f"{v/10:.1f}:1"))
            else:
                slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            row.addWidget(val_lbl)
            comp_layout.addLayout(row)
            self._comp_sliders[key] = slider
        layout.addWidget(comp_group)

        # Buttons
        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._preview)
        btn_row.addWidget(preview_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("primaryBtn")
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _on_preset_changed(self, preset_name: str):
        if preset_name == "None":
            for s in self._eq_sliders.values():
                s.setValue(0)
            return
        preset = EFFECT_PRESETS.get(preset_name, {})
        eq_bands = preset.get("eq", [])
        for band in eq_bands:
            if band.frequency in self._eq_sliders:
                self._eq_sliders[band.frequency].setValue(int(band.gain_db * 10))
        comp = preset.get("compressor")
        if comp:
            for key, slider in self._comp_sliders.items():
                if key in comp:
                    val = comp[key]
                    if key == "ratio":
                        slider.setValue(int(val * 10))
                    else:
                        slider.setValue(int(val))

    def _get_processed(self) -> np.ndarray:
        eq_bands = [
            EQBand(freq, slider.value() / 10.0, 1.0)
            for freq, slider in self._eq_sliders.items()
        ]
        comp_params = {
            "threshold_db": float(self._comp_sliders["threshold_db"].value()),
            "ratio": self._comp_sliders["ratio"].value() / 10.0,
            "attack_ms": float(self._comp_sliders["attack_ms"].value()),
            "release_ms": float(self._comp_sliders["release_ms"].value()),
        }
        preset = self._preset_combo.currentText()
        if preset != "None":
            return apply_preset(self._audio, self._sr, preset,
                                custom_eq=eq_bands, custom_comp=comp_params)
        # Apply custom EQ + compressor
        result = apply_eq(self._audio, self._sr, eq_bands)
        result = apply_compressor(result, self._sr, **comp_params)
        return result

    def _preview(self):
        processed = self._get_processed()
        self._engine.play_preview(processed, self._sr)

    def _apply(self):
        self._result = self._get_processed()
        self.accept()

    @property
    def result(self) -> np.ndarray | None:
        return self._result


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_DISPLAY_NAME} - AI Stem Separator")
        self.setMinimumSize(960, 680)
        self.setAcceptDrops(True)
        icon_path = get_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        self._input_file: str | None = None
        self._original_input_path: str | None = None
        self._output_dir: str | None = None
        self._output_files: list[str] = []
        self._worker: SeparationWorker | None = None
        self._batch_worker: BatchSeparationWorker | None = None
        self._analysis_worker: AudioAnalysisWorker | None = None
        self._settings = get_app_settings()
        self._settings.apply_runtime_environment()
        self._acceleration_report = None

        self._engine = AudioEngine(self)
        self._engine.position_changed.connect(self._on_position_changed)
        self._engine.playback_started.connect(self._on_engine_started)
        self._engine.playback_finished.connect(self._on_engine_stopped)
        self._engine.playback_paused.connect(self._on_engine_stopped)

        self._stem_widgets: list[StemTrackWidget] = []
        self._original_track_widget: StemTrackWidget | None = None
        self._effects_dialog: LayerEffectsDialog | None = None
        self._suppress_effect_dialog_discard = False
        self._layer_effect_states: dict[str, dict[str, dict[str, float | bool]]] = {}
        self._preview_layer_effect_states: dict[str, dict[str, dict[str, float | bool]]] = {}
        self._layer_labels: dict[str, str] = {}
        self._effects_preview_session = 0
        self._effects_preview_request_counter = 0
        self._effects_preview_latest_requests: dict[str, int] = {}
        self._effects_preview_processor = LiveEffectsPreviewProcessor(self)
        self._effects_preview_processor.result_ready.connect(self._on_effect_preview_ready)
        self._effects_preview_processor.error.connect(self._on_effect_preview_error)
        self._effects_preview_processor.busy_changed.connect(self._on_effect_preview_busy_changed)
        self._split_clip_action: QAction | None = None
        self._split_clip_from_file_action: QAction | None = None
        self._extract_audio_action: QAction | None = None
        self._extract_audio_from_file_action: QAction | None = None
        self._hsplitter: QSplitter | None = None
        self._left_scroll: QScrollArea | None = None
        self._initial_geometry_applied = False
        self._screen_change_connected = False

        self.setStyleSheet(DARK_STYLE)
        self._build_menu_bar()
        self._build_ui()
        self._build_status_bar()
        self._refresh_output_label()
        self._update_gpu_info()

        self._log_viewer.install()

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_screen_tracking()
        if not self._initial_geometry_applied:
            self._apply_adaptive_window_geometry(force=True, center=True)
            self._initial_geometry_applied = True
            QTimer.singleShot(0, self._apply_splitter_defaults)

    def closeEvent(self, event):
        if self._effects_dialog is not None:
            self._suppress_effect_dialog_discard = True
            self._effects_dialog.close()
            self._suppress_effect_dialog_discard = False
            self._effects_dialog = None
        self._effects_preview_processor.stop()
        self._engine.stop()
        self._log_viewer.uninstall()
        super().closeEvent(event)

    # ── Menu Bar ──

    def _build_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        self._add_action(file_menu, "&Open Audio File...", self._open_file, "Ctrl+O")
        self._add_action(file_menu, "Open &Folder (Batch)...", self._open_batch_folder)
        file_menu.addSeparator()
        self._add_action(file_menu, "Set &Output Directory...", self._set_output_dir)
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", self.close, "Ctrl+Q")

        edit_menu = menu_bar.addMenu("&Edit")
        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(lambda checked=False: self._open_settings("general"))
        edit_menu.addAction(settings_action)

        view_menu = menu_bar.addMenu("&View")
        self._toggle_log_action = QAction("Show &Console Log", self)
        self._toggle_log_action.setShortcut("Ctrl+L")
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(False)
        self._toggle_log_action.triggered.connect(self._toggle_log_panel)
        view_menu.addAction(self._toggle_log_action)

        tools_menu = menu_bar.addMenu("&Tools")
        self._split_clip_action = QAction("Split Clip", self)
        self._split_clip_action.setEnabled(False)
        self._split_clip_action.triggered.connect(self._open_split_dialog)
        tools_menu.addAction(self._split_clip_action)
        self._split_clip_from_file_action = QAction("Split Clip From File...", self)
        self._split_clip_from_file_action.triggered.connect(self._open_split_dialog_from_file)
        tools_menu.addAction(self._split_clip_from_file_action)
        tools_menu.addSeparator()
        self._extract_audio_action = QAction("Extract Audio", self)
        self._extract_audio_action.setEnabled(False)
        self._extract_audio_action.triggered.connect(self._extract_audio_for_loaded_video)
        tools_menu.addAction(self._extract_audio_action)
        self._extract_audio_from_file_action = QAction("Extract Audio From Video...", self)
        self._extract_audio_from_file_action.triggered.connect(self._extract_audio_from_video_file)
        tools_menu.addAction(self._extract_audio_from_file_action)

        help_menu = menu_bar.addMenu("&Help")
        self._add_action(help_menu, "&About", self._show_about)
        self._add_action(help_menu, "&GPU Info", self._show_gpu_info)

    def _add_action(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)

    @staticmethod
    def _make_help_label(text: str, *, font_size: int = 10, panel: bool = False) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        style = f"color: #a6adc8; font-size: {font_size}px; padding: 2px;"
        if panel:
            style += " background-color: #181825; border-radius: 4px; padding: 4px;"
        label.setStyleSheet(style)
        return label

    @staticmethod
    def _set_slider_label(slider: QSlider, label: QLabel, formatter) -> None:
        label.setText(formatter(slider.value()))
        slider.valueChanged.connect(lambda value: label.setText(formatter(value)))

    def _add_slider_row(
        self,
        layout: QVBoxLayout,
        label_text: str,
        *,
        minimum: int,
        maximum: int,
        value: int,
        formatter,
        tooltip: str,
    ) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setToolTip(tooltip)
        value_label = QLabel("")
        self._set_slider_label(slider, value_label, formatter)
        row.addWidget(slider, stretch=1)
        row.addWidget(value_label)
        layout.addLayout(row)
        return slider, value_label

    def _build_dereverb_params_group(self, attr_prefix: str) -> QGroupBox:
        group = QGroupBox("De-Echo/Roformer Parameters")
        group.setToolTip(
            "Advanced settings for the BS-Roformer de-echo/de-reverb stage.\n"
            "These are passed to audio-separator as MDXC/Roformer parameters."
        )
        layout = QVBoxLayout(group)

        segment_slider, _ = self._add_slider_row(
            layout,
            "Segment size:",
            minimum=32,
            maximum=512,
            value=256,
            formatter=lambda value: str(value),
            tooltip=(
                "Size of each Roformer processing segment.\n\n"
                "Lower values use less VRAM but give the model less room context.\n"
                "Higher values can improve reverb removal on long room tails but use more memory.\n"
                "256 is the balanced default."
            ),
        )
        overlap_slider, _ = self._add_slider_row(
            layout,
            "Overlap:",
            minimum=1,
            maximum=32,
            value=8,
            formatter=lambda value: str(value),
            tooltip=(
                "Number of overlapping Roformer segments at each boundary.\n\n"
                "Low values (1-4): faster, but boundary artifacts are more likely.\n"
                "8: balanced default.\n"
                "12-16: cleaner de-echo/de-reverb on long or difficult recordings.\n"
                "24-32: much slower with diminishing returns."
            ),
        )
        batch_slider, _ = self._add_slider_row(
            layout,
            "Batch size:",
            minimum=1,
            maximum=16,
            value=1,
            formatter=lambda value: str(value),
            tooltip=(
                "How many Roformer segments are processed at once.\n\n"
                "Changes speed and VRAM use only; it does not change output quality.\n"
                "Use 1 for maximum compatibility. Try 2-4 on GPUs with spare VRAM."
            ),
        )
        pitch_slider, _ = self._add_slider_row(
            layout,
            "Pitch shift:",
            minimum=-12,
            maximum=12,
            value=0,
            formatter=lambda value: f"{value:+d}",
            tooltip=(
                "Pitch-shifts audio during Roformer processing, then shifts it back.\n\n"
                "0 is the normal/default setting.\n"
                "Try +2 for very low-pitched voices or bass-heavy recordings.\n"
                "Try -2 for very bright or high-pitched recordings.\n"
                "Extreme values are slower and can make reverb removal less natural."
            ),
        )

        layout.addWidget(self._make_help_label(
            "Overlap is the setting the Studio Sound tip refers to. "
            "Increase it when reverb removal leaves repeating boundary artifacts; "
            "lower it when you need faster processing or hit GPU memory limits. "
            "Leave pitch shift at 0 unless the model seems to miss very low or very bright speech.",
            font_size=9,
            panel=True,
        ))

        setattr(self, f"_{attr_prefix}_dereverb_segment_slider", segment_slider)
        setattr(self, f"_{attr_prefix}_dereverb_overlap_slider", overlap_slider)
        setattr(self, f"_{attr_prefix}_dereverb_batch_slider", batch_slider)
        setattr(self, f"_{attr_prefix}_dereverb_pitch_slider", pitch_slider)
        return group

    def _build_mdx_enhance_params_group(self) -> QGroupBox:
        group = QGroupBox("MDX-Net Parameters")
        group.setToolTip("Advanced settings for the MDX-Net vocal/speech isolation stage.")
        layout = QVBoxLayout(group)

        self._mdx_segment_slider, _ = self._add_slider_row(
            layout,
            "Segment size:",
            minimum=32,
            maximum=512,
            value=256,
            formatter=lambda value: str(value),
            tooltip=(
                "Size of each MDX-Net audio segment.\n\n"
                "Lower values use less VRAM but can sound less coherent.\n"
                "Higher values can improve isolation quality but cost more memory.\n"
                "256 is the balanced default."
            ),
        )
        self._mdx_overlap_slider, _ = self._add_slider_row(
            layout,
            "Overlap:",
            minimum=1,
            maximum=95,
            value=25,
            formatter=lambda value: f"{value / 100:.2f}",
            tooltip=(
                "How much adjacent MDX-Net chunks overlap before crossfading.\n\n"
                "0.10: faster preview setting.\n"
                "0.25: balanced default.\n"
                "0.50: higher quality for noisy or long recordings.\n"
                "Above 0.75 is usually much slower without much benefit."
            ),
        )
        self._mdx_batch_slider, _ = self._add_slider_row(
            layout,
            "Batch size:",
            minimum=1,
            maximum=16,
            value=1,
            formatter=lambda value: str(value),
            tooltip=(
                "How many MDX-Net segments are processed at once.\n\n"
                "Changes speed and VRAM use only; it does not change output quality.\n"
                "Use 1 for safest processing. Try 2-4 on GPUs with spare VRAM."
            ),
        )
        layout.addWidget(self._make_help_label(
            "Fixed for this enhancement preset: hop length 1024. "
            "The denoise checkbox below controls audio-separator's MDX denoise pass.",
            font_size=9,
            panel=True,
        ))
        return group

    # ── Main UI ──

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(12, 8, 12, 8)

        # Vertical splitter: top (main area) | bottom (log panel)
        self._vsplitter = QSplitter(Qt.Orientation.Vertical)
        self._vsplitter.setChildrenCollapsible(True)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self._hsplitter = QSplitter(Qt.Orientation.Horizontal)
        self._hsplitter.setChildrenCollapsible(False)
        self._hsplitter.setHandleWidth(8)

        # ── Left Panel: Controls ──
        left_panel = self._build_left_panel()
        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._left_scroll.setMinimumWidth(320)
        self._left_scroll.setWidget(left_panel)
        self._hsplitter.addWidget(self._left_scroll)

        # ── Right Panel: Content ──
        right_panel = self._build_right_panel()
        self._hsplitter.addWidget(right_panel)

        self._hsplitter.setStretchFactor(0, 0)
        self._hsplitter.setStretchFactor(1, 1)
        top_layout.addWidget(self._hsplitter)

        self._vsplitter.addWidget(top_widget)

        # Log panel (hidden by default)
        self._log_viewer = LogViewerWidget()
        self._log_viewer.setVisible(False)
        self._vsplitter.addWidget(self._log_viewer)
        self._vsplitter.setSizes([750, 0])

        main_layout.addWidget(self._vsplitter)
        self._on_sep_subcategory_changed(0)  # initialise to Standard Models

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        # Input file
        input_group = QGroupBox("Input Audio")
        input_layout = QVBoxLayout(input_group)
        self._input_label = QLabel("No file selected (drag & drop or click Open)")
        self._input_label.setWordWrap(True)
        self._input_label.setStyleSheet("color: #a6adc8; padding: 4px;")
        open_btn = QPushButton("Open Audio File...")
        open_btn.clicked.connect(self._open_file)
        input_layout.addWidget(self._input_label)
        input_layout.addWidget(open_btn)
        layout.addWidget(input_group)

        # ── Main tab widget: Separation | Enhancement ──
        self._main_tabs = QTabWidget()
        self._main_tabs.setDocumentMode(True)
        layout.addWidget(self._main_tabs)

        # ── Separation tab ──
        sep_tab = QWidget()
        sep_tab_layout = QVBoxLayout(sep_tab)
        sep_tab_layout.setContentsMargins(4, 4, 4, 4)
        sep_tab_layout.setSpacing(6)
        self._main_tabs.addTab(sep_tab, "Separation")

        # Sub-category: Standard Models / SAM-Audio
        sub_cat_row = QHBoxLayout()
        sub_cat_row.addWidget(QLabel("Type:"))
        self._sub_cat_combo = QComboBox()
        self._sub_cat_combo.addItems(["Standard Models", "SAM-Audio"])
        self._sub_cat_combo.currentIndexChanged.connect(self._on_sep_subcategory_changed)
        sub_cat_row.addWidget(self._sub_cat_combo, stretch=1)
        sep_tab_layout.addLayout(sub_cat_row)

        # Keep model_group as a plain widget (no title) inside the tab
        model_group = QWidget()
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(0, 0, 0, 0)
        sep_tab_layout.addWidget(model_group)

        self._model_tree = QTreeWidget()
        self._model_tree.setHeaderHidden(True)
        self._model_tree.setRootIsDecorated(False)
        self._model_tree.setMaximumHeight(150)
        self._model_tree.currentItemChanged.connect(self._on_model_selected)
        model_layout.addWidget(self._model_tree)

        self._model_desc = QLabel("")
        self._model_desc.setWordWrap(True)
        self._model_desc.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 4px;")
        model_layout.addWidget(self._model_desc)

        # ── Model Parameters (architecture-specific, shown for standard models) ──
        self._model_params_group = QGroupBox("Model Parameters")
        mp_layout = QVBoxLayout(self._model_params_group)

        # Preset row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._param_preset_combo = QComboBox()
        self._param_preset_combo.addItems(["Default", "High Quality", "Fast", "Low VRAM"])
        self._param_preset_combo.currentTextChanged.connect(self._on_param_preset_changed)
        self._param_preset_combo.setToolTip(
            "Default: balanced quality and speed\n"
            "High Quality: best results, slower processing\n"
            "Fast: quick results, lower quality\n"
            "Low VRAM: reduced memory usage for GPUs with <8GB")
        preset_row.addWidget(self._param_preset_combo, stretch=1)
        mp_layout.addLayout(preset_row)

        # -- Segment Size --
        self._param_segment_row = QWidget()
        seg_row = QHBoxLayout(self._param_segment_row)
        seg_row.setContentsMargins(0, 0, 0, 0)
        seg_row.addWidget(QLabel("Segment size:"))
        self._param_segment_slider = QSlider(Qt.Orientation.Horizontal)
        self._param_segment_slider.setRange(32, 512)
        self._param_segment_slider.setValue(256)
        self._param_segment_label = QLabel("256")
        self._param_segment_slider.valueChanged.connect(
            lambda v: self._param_segment_label.setText(str(v)))
        self._param_segment_slider.setToolTip(
            "Size of each audio chunk fed to the model.\n\n"
            "Too small (32–64): Model loses musical context between notes/phrases.\n"
            "  Result: 'choppy' artifacts, poor bass separation, disconnected harmonics.\n"
            "Too large (384–512): Better quality but VRAM usage rises sharply.\n"
            "  8GB GPU: use ≤ 256. 16GB GPU: can try 512.\n\n"
            "Default 256 — good balance for 8GB+ VRAM.")
        seg_row.addWidget(self._param_segment_slider, stretch=1)
        seg_row.addWidget(self._param_segment_label)
        mp_layout.addWidget(self._param_segment_row)

        # -- Overlap (float, for MDX/Demucs) --
        self._param_overlap_row = QWidget()
        ovl_row = QHBoxLayout(self._param_overlap_row)
        ovl_row.setContentsMargins(0, 0, 0, 0)
        ovl_row.addWidget(QLabel("Overlap:"))
        self._param_overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self._param_overlap_slider.setRange(1, 95)
        self._param_overlap_slider.setValue(25)
        self._param_overlap_label = QLabel("0.25")
        self._param_overlap_slider.valueChanged.connect(
            lambda v: self._param_overlap_label.setText(f"{v / 100:.2f}"))
        self._param_overlap_slider.setToolTip(
            "How much adjacent chunks overlap before crossfading.\n\n"
            "Too low (0.01–0.10): Hard transitions between chunks.\n"
            "  Result: audible 'click' or 'fade' artifacts every 30–60 seconds.\n"
            "Too high (0.75–0.95): Chunks overlap excessively — processing 3–4× slower.\n"
            "  Rarely worth it above 0.50.\n\n"
            "0.25 = default (good balance).\n"
            "0.50 = high quality for long recordings or problem audio.\n"
            "Range: 0.01–0.95")
        ovl_row.addWidget(self._param_overlap_slider, stretch=1)
        ovl_row.addWidget(self._param_overlap_label)
        mp_layout.addWidget(self._param_overlap_row)

        # -- Overlap (integer, for MDXC/Roformer) --
        ovli_row = QHBoxLayout()
        ovli_row.setContentsMargins(0, 0, 0, 0)
        ovli_row.addWidget(QLabel("Overlap:"))
        self._param_overlap_int_slider = QSlider(Qt.Orientation.Horizontal)
        self._param_overlap_int_slider.setRange(1, 32)
        self._param_overlap_int_slider.setValue(8)
        self._param_overlap_int_label = QLabel("8")
        self._param_overlap_int_slider.valueChanged.connect(
            lambda v: self._param_overlap_int_label.setText(str(v)))
        self._param_overlap_int_slider.setToolTip(
            "Number of overlapping segments at each chunk boundary.\n\n"
            "Too low (1–3): Boundary artifacts between segments.\n"
            "  Result: repeating 'blip' or 'cut' sound at regular intervals.\n"
            "Too high (24–32): Exponentially slower — diminishing returns above 16.\n"
            "  Processing time can be 3–4× longer.\n\n"
            "8 = default. 12–16 = high quality for long recordings or reverb removal.\n"
            "Range: 1–32")
        ovli_row.addWidget(self._param_overlap_int_slider, stretch=1)
        ovli_row.addWidget(self._param_overlap_int_label)
        self._param_overlap_int_row = QWidget()
        self._param_overlap_int_row.setLayout(ovli_row)
        mp_layout.addWidget(self._param_overlap_int_row)

        # -- Batch Size --
        self._param_batch_row = QWidget()
        bs_row = QHBoxLayout(self._param_batch_row)
        bs_row.setContentsMargins(0, 0, 0, 0)
        bs_row.addWidget(QLabel("Batch size:"))
        self._param_batch_slider = QSlider(Qt.Orientation.Horizontal)
        self._param_batch_slider.setRange(1, 16)
        self._param_batch_slider.setValue(1)
        self._param_batch_label = QLabel("1")
        self._param_batch_slider.valueChanged.connect(
            lambda v: self._param_batch_label.setText(str(v)))
        self._param_batch_slider.setToolTip(
            "Number of audio segments processed simultaneously.\n\n"
            "Has NO effect on output quality — only affects speed and VRAM usage.\n"
            "Higher = faster processing but uses proportionally more VRAM.\n\n"
            "1 = safest default (any GPU).\n"
            "2–4 = good speedup on 8GB+ VRAM.\n"
            "8–16 = only useful on 16–24GB VRAM; risk of out-of-memory errors.\n"
            "Range: 1–16")
        bs_row.addWidget(self._param_batch_slider, stretch=1)
        bs_row.addWidget(self._param_batch_label)
        mp_layout.addWidget(self._param_batch_row)

        # -- Shifts (Demucs only) --
        shifts_row = QHBoxLayout()
        shifts_row.setContentsMargins(0, 0, 0, 0)
        shifts_row.addWidget(QLabel("Shifts:"))
        self._param_shifts_slider = QSlider(Qt.Orientation.Horizontal)
        self._param_shifts_slider.setRange(0, 20)
        self._param_shifts_slider.setValue(2)
        self._param_shifts_label = QLabel("2")
        self._param_shifts_slider.valueChanged.connect(
            lambda v: self._param_shifts_label.setText(str(v)))
        self._param_shifts_slider.setToolTip(
            "Predictions made with random time shifts, then averaged together.\n\n"
            "0: Fastest — single prediction, occasional boundary artifacts.\n"
            "1–2: Good balance. Default is 2. Small quality boost, ~2× slower.\n"
            "5–10: High quality — noticeable improvement, 5–10× slower.\n"
            "Too high (15–20): Barely better than 5, processing time is excessive.\n\n"
            "Use 0 for quick previews, 2 for normal use, 5 for final export.\n"
            "Requires GPU for values > 0 (CPU = very slow).\n"
            "Range: 0–20")
        shifts_row.addWidget(self._param_shifts_slider, stretch=1)
        shifts_row.addWidget(self._param_shifts_label)
        self._param_shifts_row = QWidget()
        self._param_shifts_row.setLayout(shifts_row)
        mp_layout.addWidget(self._param_shifts_row)

        # -- Window Size (VR only) --
        ws_row = QHBoxLayout()
        ws_row.setContentsMargins(0, 0, 0, 0)
        ws_row.addWidget(QLabel("Window size:"))
        self._param_window_combo = QComboBox()
        self._param_window_combo.addItems(["320 (Best quality)", "512 (Balanced)", "1024 (Fast)"])
        self._param_window_combo.setCurrentIndex(1)
        self._param_window_combo.setToolTip(
            "STFT window size — affects frequency resolution vs time resolution.\n\n"
            "320: Highest frequency resolution → best pitch/tone separation.\n"
            "  Result: better vocal detail, more precise harmonics. ~30% slower.\n"
            "512: Balanced — good default for most cases.\n"
            "1024: Fastest, lower frequency resolution.\n"
            "  Result: slightly 'smeared' separation; vocals may bleed into instrumental.\n\n"
            "Use 320 for critical final exports, 512 for everyday use.")
        ws_row.addWidget(self._param_window_combo, stretch=1)
        self._param_window_row = QWidget()
        self._param_window_row.setLayout(ws_row)
        mp_layout.addWidget(self._param_window_row)

        # -- Aggression (VR only) --
        agg_row = QHBoxLayout()
        agg_row.setContentsMargins(0, 0, 0, 0)
        agg_row.addWidget(QLabel("Aggression:"))
        self._param_aggression_slider = QSlider(Qt.Orientation.Horizontal)
        self._param_aggression_slider.setRange(-100, 100)
        self._param_aggression_slider.setValue(5)
        self._param_aggression_label = QLabel("5")
        self._param_aggression_slider.valueChanged.connect(
            lambda v: self._param_aggression_label.setText(str(v)))
        self._param_aggression_slider.setToolTip(
            "How strongly the model extracts the primary stem.\n\n"
            "Too low (0–3): Under-extraction. Vocals bleed into instrumental.\n"
            "  Result: thin, quiet vocal stem; instrumental retains vocal echo.\n"
            "5: Standard for vocals — well-tuned default. Good starting point.\n"
            "Too high (7–15): Over-extraction. Instrumental frequencies pulled into vocal.\n"
            "  Result: muddy vocal stem, hollow/phasey instrumental.\n"
            "Negative values: Subtract the stem (creative inversion effect).\n\n"
            "Keep at 5 for vocals. Try 4–6 for instruments.\n"
            "Range: -100 to 100  |  Default: 5")
        agg_row.addWidget(self._param_aggression_slider, stretch=1)
        agg_row.addWidget(self._param_aggression_label)
        self._param_aggression_row = QWidget()
        self._param_aggression_row.setLayout(agg_row)
        mp_layout.addWidget(self._param_aggression_row)

        # -- Pitch Shift (MDXC/Roformer only) --
        ps_row = QHBoxLayout()
        ps_row.setContentsMargins(0, 0, 0, 0)
        ps_row.addWidget(QLabel("Pitch shift:"))
        self._param_pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self._param_pitch_slider.setRange(-12, 12)
        self._param_pitch_slider.setValue(0)
        self._param_pitch_label = QLabel("0")
        self._param_pitch_slider.valueChanged.connect(
            lambda v: self._param_pitch_label.setText(str(v)))
        self._param_pitch_slider.setToolTip(
            "Shifts audio pitch during processing only — output pitch is unchanged.\n\n"
            "Why use it: some models work in specific frequency bands. Pitch-shifting\n"
            "  moves audio into a range the model handles better, then shifts back.\n\n"
            "Positive (+2 to +6): Useful for very low-pitched sources (bass guitar,\n"
            "  kick drum, deep male voice). Cuts some upper harmonics.\n"
            "Negative (-2 to -6): Useful for very high-pitched sources (violin,\n"
            "  female falsetto, bright synths). Takes ~20% longer to process.\n\n"
            "0 = no shift (default). Try ±2 if separation seems incomplete or off.\n"
            "Range: -12 to +12 semitones")
        ps_row.addWidget(self._param_pitch_slider, stretch=1)
        ps_row.addWidget(self._param_pitch_label)
        self._param_pitch_row = QWidget()
        self._param_pitch_row.setLayout(ps_row)
        mp_layout.addWidget(self._param_pitch_row)

        # -- Checkboxes --
        self._param_denoise_cb = QCheckBox("Enable denoise pass")
        self._param_denoise_cb.setToolTip(
            "Run the model twice — once forward, once in reverse — then average results.\n\n"
            "Effect: reduces random artifacts and noise in the output stems.\n"
            "Cost: doubles processing time.\n\n"
            "Recommended when: source audio is noisy, lo-fi, or has background hiss.\n"
            "Skip when: clean studio recordings (no quality benefit, wastes time).")
        mp_layout.addWidget(self._param_denoise_cb)

        self._param_tta_cb = QCheckBox("Test-Time Augmentation (TTA)")
        self._param_tta_cb.setToolTip(
            "Test-Time Augmentation: runs multiple predictions with mirrored audio,\n"
            "then averages results to reduce artifacts.\n\n"
            "Quality gain: +15–20% improvement in separation clarity.\n"
            "Cost: 3–4× slower processing.\n\n"
            "Use for final exports of critical material — too slow for previews.\n"
            "Requires GPU. VR architecture only.")
        mp_layout.addWidget(self._param_tta_cb)

        self._param_high_end_cb = QCheckBox("High-end frequency processing")
        self._param_high_end_cb.setToolTip(
            "Mirror the missing high-frequency range into the output stem.\n\n"
            "Some VR models have a frequency cutoff and discard very high frequencies.\n"
            "This option reconstructs them by mirroring the top of the spectrum.\n\n"
            "When to use: output sounds dull/muffled compared to source.\n"
            "When to skip: output already has full high-frequency content.\n"
            "VR architecture only.")
        mp_layout.addWidget(self._param_high_end_cb)

        self._param_post_process_cb = QCheckBox("Post-process artifacts")
        self._param_post_process_cb.setToolTip(
            "Post-processing pass to remove vocal artifacts from the instrumental stem.\n\n"
            "Applies a threshold filter to suppress residual vocal bleed.\n\n"
            "When to use: instrumental stem still has audible vocal ghost after separation.\n"
            "Warning: can degrade the instrumental stem if applied unnecessarily.\n"
            "  Use as last resort only — try other models first.\n"
            "VR architecture only.")
        mp_layout.addWidget(self._param_post_process_cb)

        self._param_segments_enabled_cb = QCheckBox("Enable segmented processing")
        self._param_segments_enabled_cb.setChecked(True)
        self._param_segments_enabled_cb.setToolTip(
            "Process audio in segments to keep VRAM usage bounded.\n\n"
            "Enabled (default): audio is split into chunks, each processed separately.\n"
            "  Required for long files or limited VRAM (< 16GB).\n"
            "Disabled: process the entire file as one chunk — requires massive VRAM.\n"
            "  Only useful on workstation GPUs (24GB+) for very short files.\n\n"
            "Keep enabled unless you specifically need whole-file processing.\n"
            "Demucs architecture only.")
        mp_layout.addWidget(self._param_segments_enabled_cb)

        # -- Help label --
        self._param_help = QLabel("")
        self._param_help.setWordWrap(True)
        self._param_help.setStyleSheet(
            "color: #a6adc8; font-size: 10px; padding: 4px; "
            "background-color: #181825; border-radius: 4px;")
        self._param_help.setVisible(False)
        mp_layout.addWidget(self._param_help)

        self._model_params_group.setVisible(False)
        model_layout.addWidget(self._model_params_group)

        # SAM-Audio options (hidden by default, shown when SAM category selected)
        self._sam_options = QWidget()
        sam_opts_layout = QVBoxLayout(self._sam_options)
        sam_opts_layout.setContentsMargins(0, 4, 0, 0)

        # Model variant selector
        sam_opts_layout.addWidget(QLabel("Model:"))
        self._sam_model_combo = QComboBox()
        self._sam_model_combo.addItems([
            "SAM-Audio Large",
            "SAM-Audio Base",
            "SAM-Audio Small",
            "SAM-Audio Large TV",
            "SAM-Audio Small TV",
        ])
        self._sam_model_combo.setToolTip(
            "TV variants are better for correctness of target sound and visual prompting"
        )
        sam_opts_layout.addWidget(self._sam_model_combo)

        self._sam_prompt = QLineEdit()
        self._sam_prompt.setPlaceholderText("e.g. 'isolate the drums'")
        sam_opts_layout.addWidget(QLabel("Text prompt:"))
        sam_opts_layout.addWidget(self._sam_prompt)

        rerank_row = QHBoxLayout()
        rerank_row.addWidget(QLabel("Re-ranking candidates:"))
        self._sam_rerank_slider = QSlider(Qt.Orientation.Horizontal)
        self._sam_rerank_slider.setRange(1, 8)
        self._sam_rerank_slider.setValue(1)
        self._sam_rerank_label = QLabel("1")
        self._sam_rerank_slider.valueChanged.connect(self._on_rerank_changed)
        rerank_row.addWidget(self._sam_rerank_slider, stretch=1)
        rerank_row.addWidget(self._sam_rerank_label)
        sam_opts_layout.addLayout(rerank_row)

        self._vram_warning = QLabel("")
        self._vram_warning.setWordWrap(True)
        self._vram_warning.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._vram_warning.setVisible(False)
        sam_opts_layout.addWidget(self._vram_warning)

        self._sam_predict_spans = QCheckBox("Predict spans (recommended)")
        self._sam_predict_spans.setChecked(True)
        sam_opts_layout.addWidget(self._sam_predict_spans)

        # Advanced Chunking Settings (hidden until clip is long enough)
        self._chunk_group = QGroupBox("Advanced Chunking Settings")
        self._chunk_group.setVisible(False)
        chunk_group = self._chunk_group
        chunk_group.setCheckable(False)
        chunk_layout = QVBoxLayout(chunk_group)
        chunk_layout.setContentsMargins(4, 8, 4, 4)
        chunk_layout.setSpacing(4)

        self._chunk_overlap_cb = QCheckBox("Enable overlap (recommended)")
        self._chunk_overlap_cb.setChecked(True)
        chunk_layout.addWidget(self._chunk_overlap_cb)

        overlap_row = QHBoxLayout()
        overlap_row.addWidget(QLabel("Overlap:"))
        self._chunk_overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self._chunk_overlap_slider.setRange(2, 15)
        self._chunk_overlap_slider.setValue(10)
        self._chunk_overlap_label = QLabel("10s")
        self._chunk_overlap_slider.valueChanged.connect(
            lambda v: self._chunk_overlap_label.setText(f"{v}s"))
        overlap_row.addWidget(self._chunk_overlap_slider, stretch=1)
        overlap_row.addWidget(self._chunk_overlap_label)
        chunk_layout.addLayout(overlap_row)

        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Chunk:"))
        self._chunk_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self._chunk_duration_slider.setRange(10, 60)
        self._chunk_duration_slider.setValue(30)
        self._chunk_duration_label = QLabel("30s")
        self._chunk_duration_slider.valueChanged.connect(
            lambda v: self._chunk_duration_label.setText(f"{v}s"))
        duration_row.addWidget(self._chunk_duration_slider, stretch=1)
        duration_row.addWidget(self._chunk_duration_label)
        chunk_layout.addLayout(duration_row)

        crossfade_row = QHBoxLayout()
        crossfade_row.addWidget(QLabel("Crossfade:"))
        self._crossfade_combo = QComboBox()
        self._crossfade_combo.addItems(["Hann (recommended)", "Linear"])
        crossfade_row.addWidget(self._crossfade_combo, stretch=1)
        chunk_layout.addLayout(crossfade_row)

        # Wire overlap checkbox to enable/disable overlap controls
        self._chunk_overlap_cb.toggled.connect(self._chunk_overlap_slider.setEnabled)
        self._chunk_overlap_cb.toggled.connect(self._crossfade_combo.setEnabled)

        sam_opts_layout.addWidget(chunk_group)

        # Visual Prompt (video only)
        visual_group = QGroupBox("Visual Prompt (video only)")
        visual_layout = QVBoxLayout(visual_group)
        visual_layout.setContentsMargins(4, 8, 4, 4)

        visual_hint = QLabel(
            "Click on an object in a video frame to isolate its sound.\n"
            "Requires a TV model variant and Re-ranking candidates \u2265 2.\n"
            "Can be combined with a text prompt for better results.")
        visual_hint.setWordWrap(True)
        visual_hint.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        visual_layout.addWidget(visual_hint)

        visual_btn_row = QHBoxLayout()
        self._visual_prompt_btn = QPushButton("Select Object in Video...")
        self._visual_prompt_btn.clicked.connect(self._open_visual_prompt)
        self._visual_prompt_btn.setEnabled(False)
        visual_btn_row.addWidget(self._visual_prompt_btn)

        self._load_mask_btn = QPushButton("Load Existing Mask...")
        self._load_mask_btn.clicked.connect(self._load_existing_mask)
        self._load_mask_btn.setToolTip(
            "Load a previously saved mask project folder\n"
            "(must contain mask_config.json and masked_video.mp4)")
        visual_btn_row.addWidget(self._load_mask_btn)
        visual_layout.addLayout(visual_btn_row)

        self._visual_status = QLabel("No mask")
        self._visual_status.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
        visual_layout.addWidget(self._visual_status)

        self._clear_mask_btn = QPushButton("Clear Mask")
        self._clear_mask_btn.setVisible(False)
        self._clear_mask_btn.clicked.connect(self._clear_visual_mask)
        visual_layout.addWidget(self._clear_mask_btn)

        sam_opts_layout.addWidget(visual_group)
        self._masked_video_path: str | None = None

        self._sam_notice = QLabel("")
        self._sam_notice.setWordWrap(True)
        self._sam_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._sam_notice.setVisible(False)
        sam_opts_layout.addWidget(self._sam_notice)

        self._sam_options.setVisible(False)
        model_layout.addWidget(self._sam_options)

        # ── Enhancement options (shown when "Audio Enhancement" selected) ──
        self._enhance_options = QWidget()
        enhance_layout = QVBoxLayout(self._enhance_options)
        enhance_layout.setContentsMargins(0, 0, 0, 0)

        # Engine selector (top-level choice)
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("Engine:"))
        self._enhance_engine_combo = QComboBox()
        self._enhance_engine_combo.addItems([
            "Resemble-Enhance",
            "DeepFilterNet (speech/vocals)",
            "Studio Sound (AI)",
            "ClearVoice (speech enhancement)",
            "MDX-Net (vocal isolation)",
            "VoiceFixer (speech restoration)",
            "MetricGAN+ (noise suppression)",
        ])
        self._enhance_engine_combo.setToolTip(
            "Resemble-Enhance: general AI enhancement — denoises + upscales to 44.1kHz.\n"
            "  Best for: any speech or vocal recording.\n\n"
            "DeepFilterNet: speech-specific noise suppression — optimised for intelligibility.\n"
            "  Best for: podcasts, interviews, voice recordings, conference calls.\n"
            "  Output: 48kHz. Tiny model (~15MB). Much faster than Resemble-Enhance.\n\n"
            "Studio Sound (AI): one-click deep cleanup — chains DeepFilterNet noise removal\n"
            "  then De-Echo/DeReverb Roformer in a single pass.\n"
            "  Best for: recordings with both background noise AND room echo/reverb.\n"
            "  Each stage is optional — use both or just one.\n\n"
            "ClearVoice: local speech-enhancement models tuned for cleanup-first restoration.\n"
            "  Best for: town halls, conference rooms, podium mics, distant speech, and HVAC noise.\n"
            "  MossFormer2_SE_48K is the recommended default. Optional de-reverb can run after it.\n\n"
            "MDX-Net: aggressively isolates the speech/vocal foreground from the rest of the recording.\n"
            "  Best for: noisy PA recordings, crowd bleed, or buried speakers when you want extraction\n"
            "  more than natural ambience. Optional de-reverb can run after isolation.\n\n"
            "VoiceFixer: all-in-one speech restoration — removes noise, reverb, echo, and\n"
            "  clipping in a single model pass, then upsamples to 44.1kHz.\n"
            "  Best for: town halls, room recordings, heavily degraded speech.\n"
            "  Model: ~1 GB download on first use.\n\n"
            "MetricGAN+: state-of-the-art noise suppressor (PESQ 3.15).\n"
            "  Noise only — no reverb. ~100 MB model.\n"
            "  Best for: clean noise removal when reverb is not an issue.")
        engine_row.addWidget(self._enhance_engine_combo, stretch=1)
        enhance_layout.addLayout(engine_row)

        level_group = QGroupBox("Output Level")
        level_layout = QVBoxLayout(level_group)
        self._enhance_auto_level_cb = QCheckBox("Auto level")
        self._enhance_auto_level_cb.setChecked(True)
        self._enhance_auto_level_cb.setToolTip(
            "Automatically level the processed file to a consistent final output level.\n"
            "Runs after Match input loudness if both are enabled.\n"
            "This is written into the saved WAV file, not just used for playback preview.\n"
            "A safety limiter is still applied afterward to prevent clipping."
        )
        level_layout.addWidget(self._enhance_auto_level_cb)
        self._enhance_auto_level_cb.toggled.connect(self._update_enhance_level_hint)

        self._enhance_match_level_cb = QCheckBox("Match input loudness")
        self._enhance_match_level_cb.setChecked(True)
        self._enhance_match_level_cb.setToolTip(
            "Keep the enhanced output close to the original clip's average loudness.\n"
            "Runs before Auto level if both are enabled.\n"
            "Useful when an enhancement model makes the result noticeably quieter.\n"
            "A safety peak limiter is applied automatically to avoid clipping."
        )
        level_layout.addWidget(self._enhance_match_level_cb)
        self._enhance_match_level_cb.toggled.connect(self._update_enhance_level_hint)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Output gain:"))
        self._enhance_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._enhance_gain_slider.setRange(-65, 65)
        self._enhance_gain_slider.setValue(0)
        self._enhance_gain_label = QLabel("+0 dB")
        self._enhance_gain_slider.valueChanged.connect(
            lambda v: self._enhance_gain_label.setText(f"{v:+d} dB")
        )
        self._enhance_gain_slider.setToolTip(
            "Apply extra gain after enhancement.\n"
            "Positive values make the output louder, negative values make it quieter.\n"
            "This is written into the saved WAV output after auto level / loudness matching,\n"
            "and still uses clip protection."
        )
        self._enhance_gain_slider.valueChanged.connect(self._update_enhance_level_hint)
        gain_row.addWidget(self._enhance_gain_slider, stretch=1)
        gain_row.addWidget(self._enhance_gain_label)
        level_layout.addLayout(gain_row)

        self._enhance_level_hint = self._make_help_label("", font_size=10, panel=True)
        level_layout.addWidget(self._enhance_level_hint)
        self._update_enhance_level_hint()
        enhance_layout.addWidget(level_group)

        # ── Resemble-Enhance controls ──
        self._resemble_widget = QWidget()
        resemble_layout = QVBoxLayout(self._resemble_widget)
        resemble_layout.setContentsMargins(0, 4, 0, 0)

        mode_group = QGroupBox("Enhancement Mode")
        mode_layout = QVBoxLayout(mode_group)
        self._enhance_mode_combo = QComboBox()
        self._enhance_mode_combo.addItems([
            "Enhance (denoise + upscale quality)",
            "Denoise only",
        ])
        mode_layout.addWidget(self._enhance_mode_combo)

        enhance_hint = QLabel(
            "Resemble-Enhance: AI speech/vocal enhancement.\n"
            "Denoises and restores audio quality to 44.1kHz.\n"
            "Handles files of any length natively (internal chunking).")
        enhance_hint.setWordWrap(True)
        enhance_hint.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        mode_layout.addWidget(enhance_hint)
        resemble_layout.addWidget(mode_group)

        # CFM Settings (only for enhance mode)
        self._cfm_group = QGroupBox("CFM Settings")
        cfm_layout = QVBoxLayout(self._cfm_group)

        # Solver
        solver_row = QHBoxLayout()
        solver_row.addWidget(QLabel("ODE Solver:"))
        self._cfm_solver_combo = QComboBox()
        self._cfm_solver_combo.addItems(["midpoint", "rk4", "euler"])
        self._cfm_solver_combo.setToolTip(
            "ODE solver used by the Conditional Flow Matching diffusion model.\n\n"
            "midpoint — balanced quality and speed. Recommended default.\n"
            "rk4 — 4th-order Runge-Kutta. Highest quality, ~2× slower than midpoint.\n"
            "euler — simple Euler method. Fastest but noticeably lower quality.\n\n"
            "For most audio: midpoint. For critical final exports: rk4.")
        solver_row.addWidget(self._cfm_solver_combo)
        cfm_layout.addLayout(solver_row)

        # Guidance/strength
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Enhancement strength:"))
        self._cfm_guidance_slider = QSlider(Qt.Orientation.Horizontal)
        self._cfm_guidance_slider.setRange(0, 100)
        self._cfm_guidance_slider.setValue(90)
        self._cfm_guidance_label = QLabel("0.90")
        self._cfm_guidance_slider.valueChanged.connect(
            lambda v: self._cfm_guidance_label.setText(f"{v / 100:.2f}"))
        self._cfm_guidance_slider.setToolTip(
            "Conditioning strength for the Resemble enhancer (lambda).\n\n"
            "Lower values stay closer to the denoised source and are safer on music.\n"
            "0.90 is the model default used by this app.\n"
            "Higher values push restoration harder, but can create synthetic or unstable details."
        )
        strength_row.addWidget(self._cfm_guidance_slider, stretch=1)
        strength_row.addWidget(self._cfm_guidance_label)
        cfm_layout.addLayout(strength_row)

        # NFE
        nfe_row = QHBoxLayout()
        nfe_row.addWidget(QLabel("Function Evaluations:"))
        self._cfm_nfe_slider = QSlider(Qt.Orientation.Horizontal)
        self._cfm_nfe_slider.setRange(1, 128)
        self._cfm_nfe_slider.setValue(64)
        self._cfm_nfe_label = QLabel("64")
        self._cfm_nfe_slider.valueChanged.connect(
            lambda v: self._cfm_nfe_label.setText(str(v)))
        nfe_row.addWidget(self._cfm_nfe_slider, stretch=1)
        nfe_row.addWidget(self._cfm_nfe_label)
        cfm_layout.addLayout(nfe_row)
        nfe_hint = QLabel(
            "Number of neural network passes per audio chunk.\n"
            "Too low (1–16): Rough, artifacty output — diffusion not fully converged.\n"
            "32–64: Good quality range. Default 64 is recommended.\n"
            "Too high (96–128): Minimal improvement over 64, noticeably slower.")
        nfe_hint.setWordWrap(True)
        nfe_hint.setStyleSheet("color: #a6adc8; font-size: 9px;")
        cfm_layout.addWidget(nfe_hint)

        # Temperature
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Prior Temperature:"))
        self._cfm_temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._cfm_temp_slider.setRange(0, 100)
        self._cfm_temp_slider.setValue(50)
        self._cfm_temp_label = QLabel("0.50")
        self._cfm_temp_slider.valueChanged.connect(
            lambda v: self._cfm_temp_label.setText(f"{v / 100:.2f}"))
        temp_row.addWidget(self._cfm_temp_slider, stretch=1)
        temp_row.addWidget(self._cfm_temp_label)
        cfm_layout.addLayout(temp_row)
        temp_hint = QLabel(
            "Controls randomness of the diffusion prior.\n"
            "Too low (0.0–0.2): Conservative output — less enhancement, safer but duller.\n"
            "0.4–0.6: Balanced. Default 0.5 works well for most speech.\n"
            "Too high (0.8–1.0): More aggressive enhancement — may introduce instability\n"
            "  or hallucinated sounds on very noisy audio.")
        temp_hint.setWordWrap(True)
        temp_hint.setStyleSheet("color: #a6adc8; font-size: 9px;")
        cfm_layout.addWidget(temp_hint)

        resemble_layout.addWidget(self._cfm_group)

        # Denoise checkbox
        self._denoise_before_cb = QCheckBox("Denoise before enhancement")
        self._denoise_before_cb.setToolTip(
            "Run a denoising pass before the full enhancement.\n"
            "Useful when the source has heavy background noise.\n"
            "Adds processing time but can improve final quality on noisy recordings.")
        resemble_layout.addWidget(self._denoise_before_cb)

        resemble_layout.addWidget(self._make_help_label(
            "Parameters shown for Resemble-Enhance: Mode chooses full enhance vs denoise only; "
            "Denoise before enhancement runs an extra cleanup pass first; ODE solver, "
            "Function evaluations, Enhancement strength, and Prior temperature control the full "
            "CFM enhancer. Full enhancement writes a restored/upscaled output; denoise only skips "
            "the CFM controls.",
            font_size=10,
            panel=True,
        ))

        # Toggle CFM settings visibility based on mode
        self._enhance_mode_combo.currentIndexChanged.connect(self._on_enhance_mode_changed)

        # Install notice if not available
        self._resemble_notice = QLabel("")
        self._resemble_notice.setWordWrap(True)
        self._resemble_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._resemble_notice.setVisible(False)
        resemble_layout.addWidget(self._resemble_notice)

        enhance_layout.addWidget(self._resemble_widget)

        # ── DeepFilterNet controls ──
        self._deepfilter_widget = QWidget()
        df_layout = QVBoxLayout(self._deepfilter_widget)
        df_layout.setContentsMargins(0, 4, 0, 0)

        df_info = QLabel(
            "DeepFilterNet: neural speech enhancement.\n"
            "Best for: podcasts, interviews, voice recordings,\n"
            "conference calls, any speech in noisy/reverberant spaces.\n"
            "Output: 48kHz WAV. Model: ~15MB (auto-downloaded).")
        df_info.setWordWrap(True)
        df_info.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        df_layout.addWidget(df_info)

        # Attenuation limit
        atten_group = QGroupBox("Attenuation Limit (dB)")
        atten_layout = QVBoxLayout(atten_group)
        atten_row = QHBoxLayout()
        self._df_atten_slider = QSlider(Qt.Orientation.Horizontal)
        self._df_atten_slider.setRange(0, 100)
        self._df_atten_slider.setValue(70)
        self._df_atten_label = QLabel("70 dB")
        self._df_atten_slider.valueChanged.connect(
            lambda v: self._df_atten_label.setText(f"{v} dB"))
        self._df_atten_slider.setToolTip(
            "How aggressively to suppress non-speech noise.\n\n"
            "Too low (0–6 dB): Almost no effect — noise barely reduced.\n"
            "18–30 dB: Gentle suppression — preserves natural room ambience.\n"
            "  Good for music or when slight background noise is acceptable.\n"
            "70 dB: Recommended default — strong suppression, speech preserved.\n"
            "  Ideal for podcasts, interviews, voice recordings.\n"
            "Too high (90–100 dB): Maximum suppression — can introduce metallic\n"
            "  or 'underwater' artifacts on complex audio or music.\n\n"
            "Start at 70. Lower to 30–40 if artifacts appear on music/instruments.")
        atten_row.addWidget(self._df_atten_slider, stretch=1)
        atten_row.addWidget(self._df_atten_label)
        atten_layout.addLayout(atten_row)
        df_layout.addWidget(atten_group)

        # Post-filter checkbox
        self._df_post_filter_cb = QCheckBox("Post-filter (extra noise reduction pass)")
        self._df_post_filter_cb.setToolTip(
            "Applies an additional minor noise reduction pass after the main model.\n"
            "Can remove residual noise at the cost of a small amount of speech naturalness.\n"
            "Only recommended when default output still has audible background noise.")
        df_layout.addWidget(self._df_post_filter_cb)

        df_layout.addWidget(self._make_help_label(
            "Parameters shown for DeepFilterNet: Attenuation limit controls how much non-speech "
            "energy can be reduced; Post-filter asks DeepFilterNet for an extra cleanup pass. "
            "Long files are processed internally in 12 second chunks with 1 second overlap; "
            "that overlap is a stability/crossfade detail, not the Studio Sound Roformer overlap.",
            font_size=10,
            panel=True,
        ))

        self._deepfilter_notice = QLabel("")
        self._deepfilter_notice.setWordWrap(True)
        self._deepfilter_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._deepfilter_notice.setVisible(False)
        df_layout.addWidget(self._deepfilter_notice)

        self._deepfilter_widget.setVisible(False)
        enhance_layout.addWidget(self._deepfilter_widget)

        # ── Studio Sound controls ──
        self._studio_sound_widget = QWidget()
        ss_layout = QVBoxLayout(self._studio_sound_widget)
        ss_layout.setContentsMargins(0, 4, 0, 0)

        ss_info = QLabel(
            "One-click AI cleanup: chains noise removal + echo/reverb removal.\n"
            "Best for: voice memos, interviews, room recordings, podcast raw audio.\n"
            "Both stages are optional — disable either if not needed.")
        ss_info.setWordWrap(True)
        ss_info.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        ss_layout.addWidget(ss_info)

        ss_stages_group = QGroupBox("Processing Stages")
        ss_stages_layout = QVBoxLayout(ss_stages_group)

        # Stage 1: Noise removal
        self._ss_noise_cb = QCheckBox("Remove background noise (DeepFilterNet)")
        self._ss_noise_cb.setChecked(True)
        self._ss_noise_cb.setToolTip(
            "Stage 1: DeepFilterNet neural noise suppression.\n"
            "Removes hiss, HVAC noise, traffic, keyboard clicks, and other background sounds.\n"
            "Output: 48kHz WAV. Runs before reverb removal.")
        ss_stages_layout.addWidget(self._ss_noise_cb)

        # Noise attenuation slider (only visible when noise CB checked)
        self._ss_atten_row = QWidget()
        ss_atten_layout = QHBoxLayout(self._ss_atten_row)
        ss_atten_layout.setContentsMargins(16, 0, 0, 0)
        ss_atten_layout.addWidget(QLabel("Noise attenuation:"))
        self._ss_atten_slider = QSlider(Qt.Orientation.Horizontal)
        self._ss_atten_slider.setRange(0, 100)
        self._ss_atten_slider.setValue(70)
        self._ss_atten_label = QLabel("70 dB")
        self._ss_atten_slider.valueChanged.connect(
            lambda v: self._ss_atten_label.setText(f"{v} dB"))
        self._ss_atten_slider.setToolTip(
            "How aggressively to suppress non-speech noise (same as DeepFilterNet engine).\n"
            "70 dB is the recommended default for voice recordings.")
        ss_atten_layout.addWidget(self._ss_atten_slider, stretch=1)
        ss_atten_layout.addWidget(self._ss_atten_label)
        ss_stages_layout.addWidget(self._ss_atten_row)

        self._ss_post_filter_cb = QCheckBox("Post-filter noise stage")
        self._ss_post_filter_cb.setToolTip(
            "Use DeepFilterNet's additional post-filter on the noise-removal stage.\n"
            "Can remove residual background noise, but may make speech less natural."
        )
        ss_stages_layout.addWidget(self._ss_post_filter_cb)

        # Toggle noise-stage controls when noise checkbox toggled
        self._ss_noise_cb.toggled.connect(self._ss_atten_row.setVisible)
        self._ss_noise_cb.toggled.connect(self._ss_post_filter_cb.setVisible)

        # Stage 2: Reverb removal
        self._ss_reverb_cb = QCheckBox("Remove room echo / reverb (De-Echo Roformer)")
        self._ss_reverb_cb.setChecked(True)
        self._ss_reverb_cb.setToolTip(
            "Stage 2: De-Echo/DeReverb BS-Roformer model via audio-separator.\n"
            "Removes room reflections, echo, and reverb from the audio.\n"
            "Model auto-downloaded on first use (~200MB). Runs after noise removal.\n"
            "Tip: increase Overlap to 12-16 in De-Echo/Roformer Parameters for cleaner results.")
        ss_stages_layout.addWidget(self._ss_reverb_cb)

        self._ss_dereverb_group = self._build_dereverb_params_group("ss")
        self._ss_reverb_cb.toggled.connect(self._ss_dereverb_group.setVisible)
        ss_stages_layout.addWidget(self._ss_dereverb_group)

        ss_stages_layout.addWidget(self._make_help_label(
            "Parameters shown for Studio Sound: Noise removal uses DeepFilterNet attenuation; "
            "Post-filter adds DeepFilterNet's extra cleanup pass; Reverb removal uses the "
            "Roformer segment, overlap, batch, and pitch controls above. Output level is applied "
            "once after the selected stages finish.",
            font_size=10,
            panel=True,
        ))

        ss_layout.addWidget(ss_stages_group)

        self._studio_sound_notice = QLabel("")
        self._studio_sound_notice.setWordWrap(True)
        self._studio_sound_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._studio_sound_notice.setVisible(False)
        ss_layout.addWidget(self._studio_sound_notice)

        self._studio_sound_widget.setVisible(False)
        enhance_layout.addWidget(self._studio_sound_widget)

        # ── ClearVoice controls ──
        self._clearvoice_widget = QWidget()
        cv_layout = QVBoxLayout(self._clearvoice_widget)
        cv_layout.setContentsMargins(0, 4, 0, 0)

        cv_info = QLabel(
            "Cleanup-first local speech enhancement for difficult spoken-word recordings.\n"
            "Best for: town halls, conference rooms, distant speech, podium mics, and HVAC noise.\n"
            "Tip: the 16 kHz models can save back at the original higher sample rate for compatibility,\n"
            "but that does not recreate true high-frequency detail.")
        cv_info.setWordWrap(True)
        cv_info.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        cv_layout.addWidget(cv_info)

        cv_model_group = QGroupBox("Model")
        cv_model_layout = QVBoxLayout(cv_model_group)
        cv_model_row = QHBoxLayout()
        cv_model_row.addWidget(QLabel("ClearVoice model:"))
        self._cv_model_combo = QComboBox()
        self._cv_model_combo.addItem("MossFormer2_SE_48K (Recommended)", "MossFormer2_SE_48K")
        self._cv_model_combo.addItem("FRCRN_SE_16K", "FRCRN_SE_16K")
        self._cv_model_combo.setToolTip(
            "MossFormer2_SE_48K: best overall local cleanup quality for town halls and room speech.\n"
            "FRCRN_SE_16K: lighter 16 kHz enhancement model."
        )
        cv_model_row.addWidget(self._cv_model_combo, stretch=1)
        cv_model_layout.addLayout(cv_model_row)
        cv_layout.addWidget(cv_model_group)

        self._cv_reverb_cb = QCheckBox("Remove room echo / reverb after enhancement")
        self._cv_reverb_cb.setChecked(True)
        self._cv_reverb_cb.setToolTip(
            "Runs the same BS-Roformer De-Echo/DeReverb stage used by Studio Sound after ClearVoice.\n"
            "Recommended for town halls and other roomy speech recordings."
        )
        cv_layout.addWidget(self._cv_reverb_cb)
        self._cv_dereverb_group = self._build_dereverb_params_group("cv")
        self._cv_reverb_cb.toggled.connect(self._cv_dereverb_group.setVisible)
        cv_layout.addWidget(self._cv_dereverb_group)

        cv_layout.addWidget(self._make_help_label(
            "Parameters shown for ClearVoice: Model chooses the speech enhancement checkpoint; "
            "the optional de-reverb stage uses the Roformer segment, overlap, batch, and pitch controls. "
            "MossFormer2_SE_48K preserves a 48 kHz model path; FRCRN_SE_16K is lighter and runs at 16 kHz.",
            font_size=10,
            panel=True,
        ))

        self._clearvoice_notice = QLabel("")
        self._clearvoice_notice.setWordWrap(True)
        self._clearvoice_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._clearvoice_notice.setVisible(False)
        cv_layout.addWidget(self._clearvoice_notice)

        self._clearvoice_widget.setVisible(False)
        enhance_layout.addWidget(self._clearvoice_widget)

        # ── MDX-Net controls ──
        self._mdxnet_widget = QWidget()
        mdx_layout = QVBoxLayout(self._mdxnet_widget)
        mdx_layout.setContentsMargins(0, 4, 0, 0)

        mdx_info = QLabel(
            "Speech-forward cleanup using MDX-Net vocal isolation.\n"
            "Best for: town halls, podium mics, PA systems, crowd bleed, and noisy room recordings.\n"
            "Outputs two stems in the app: Speaker / voice and Background / bleed / room.\n"
            "This extracts the likely speaker foreground first, so it can sound more isolated but less natural\n"
            "than ClearVoice on already-clean material.")
        mdx_info.setWordWrap(True)
        mdx_info.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        mdx_layout.addWidget(mdx_info)

        mdx_model_group = QGroupBox("Model")
        mdx_model_layout = QVBoxLayout(mdx_model_group)
        mdx_model_row = QHBoxLayout()
        mdx_model_row.addWidget(QLabel("MDX-Net model:"))
        self._mdx_model_combo = QComboBox()
        for model_name, metadata in MDXNET_ENHANCEMENT_PRESETS.items():
            self._mdx_model_combo.addItem(metadata["label"], model_name)
        self._mdx_model_combo.setToolTip(
            "Kim Vocal 2: fastest and the best general starting point for spoken-word isolation.\n"
            "Kuielab Vocals: lighter alternative when Kim Vocal 2 sounds too carved out.\n"
            "UVR MDX-NET Inst HQ 4: more aggressive separation when the speaker is buried."
        )
        mdx_model_row.addWidget(self._mdx_model_combo, stretch=1)
        mdx_model_layout.addLayout(mdx_model_row)
        mdx_layout.addWidget(mdx_model_group)

        self._mdx_params_group = self._build_mdx_enhance_params_group()
        mdx_layout.addWidget(self._mdx_params_group)

        self._mdx_denoise_cb = QCheckBox("Enable MDX-Net denoise pass")
        self._mdx_denoise_cb.setChecked(True)
        self._mdx_denoise_cb.setToolTip(
            "Runs the model with its denoise option enabled.\n"
            "Usually helps on noisy speech recordings, but can slightly thin very clean audio."
        )
        mdx_layout.addWidget(self._mdx_denoise_cb)

        self._mdx_reverb_cb = QCheckBox("Remove room echo / reverb after isolation")
        self._mdx_reverb_cb.setChecked(True)
        self._mdx_reverb_cb.setToolTip(
            "Runs the same BS-Roformer De-Echo/DeReverb stage used by Studio Sound after MDX-Net.\n"
            "Applied to the Speaker / voice stem only. Recommended for town halls and other roomy speech recordings."
        )
        mdx_layout.addWidget(self._mdx_reverb_cb)
        self._mdx_dereverb_group = self._build_dereverb_params_group("mdx")
        self._mdx_reverb_cb.toggled.connect(self._mdx_dereverb_group.setVisible)
        mdx_layout.addWidget(self._mdx_dereverb_group)

        mdx_layout.addWidget(self._make_help_label(
            "Parameters shown for MDX-Net: Model chooses the vocal-isolation checkpoint; "
            "Segment size, overlap, batch size, and denoise control the isolation pass; "
            "the optional de-reverb stage has its own Roformer overlap, segment, batch, and pitch controls.",
            font_size=10,
            panel=True,
        ))

        self._mdxnet_notice = QLabel("")
        self._mdxnet_notice.setWordWrap(True)
        self._mdxnet_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._mdxnet_notice.setVisible(False)
        mdx_layout.addWidget(self._mdxnet_notice)

        self._mdxnet_widget.setVisible(False)
        enhance_layout.addWidget(self._mdxnet_widget)

        # ── VoiceFixer controls ──
        self._voicefixer_widget = QWidget()
        vf_layout = QVBoxLayout(self._voicefixer_widget)
        vf_layout.setContentsMargins(0, 4, 0, 0)

        vf_info = QLabel(
            "All-in-one speech restoration: removes noise, echo, reverb, and clipping\n"
            "in a single model pass, then upsamples to 44.1 kHz.\n"
            "Best for: town halls, room recordings, heavily degraded speech.\n"
            "Model: ~1 GB download on first use.")
        vf_info.setWordWrap(True)
        vf_info.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        vf_layout.addWidget(vf_info)

        vf_mode_group = QGroupBox("Restoration Mode")
        vf_mode_layout = QVBoxLayout(vf_mode_group)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._vf_mode_combo = QComboBox()
        self._vf_mode_combo.addItems([
            "0 — Full restoration (noise + reverb + super-res)",
            "1 — Aggressive restoration",
            "2 — Full restoration without upsampling",
        ])
        self._vf_mode_combo.setToolTip(
            "Mode 0: Full pass — noise removal, dereverberation, declipping, and\n"
            "  audio super-resolution to 44.1 kHz. Recommended default.\n\n"
            "Mode 1: More aggressive version of Mode 0. Stronger processing but\n"
            "  may introduce slight artifacts on very clean audio.\n\n"
            "Mode 2: Same as Mode 0 but skips the final upsampling step.\n"
            "  Output stays at the original sample rate.")
        mode_row.addWidget(self._vf_mode_combo, stretch=1)
        vf_mode_layout.addLayout(mode_row)
        vf_layout.addWidget(vf_mode_group)

        vf_layout.addWidget(self._make_help_label(
            "Parameters shown for VoiceFixer: Mode selects the restoration profile exposed by "
            "VoiceFixer. CUDA is selected automatically when available. Long files are stitched "
            "internally in 180 second chunks with 2 seconds of crossfade overlap.",
            font_size=10,
            panel=True,
        ))

        self._voicefixer_notice = QLabel("")
        self._voicefixer_notice.setWordWrap(True)
        self._voicefixer_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._voicefixer_notice.setVisible(False)
        vf_layout.addWidget(self._voicefixer_notice)

        self._voicefixer_widget.setVisible(False)
        enhance_layout.addWidget(self._voicefixer_widget)

        # ── MetricGAN+ controls ──
        self._metricgan_widget = QWidget()
        mg_layout = QVBoxLayout(self._metricgan_widget)
        mg_layout.setContentsMargins(0, 4, 0, 0)

        mg_info = QLabel(
            "State-of-the-art noise suppression (PESQ 3.15 on VoiceBank+DEMAND).\n"
            "Noise only — no reverb. ~100 MB model download on first use.\n"
            "Tip: combine with Studio Sound's reverb stage for full noise + reverb cleanup.")
        mg_info.setWordWrap(True)
        mg_info.setStyleSheet("color: #a6adc8; font-size: 10px; padding: 2px;")
        mg_layout.addWidget(mg_info)

        mg_layout.addWidget(self._make_help_label(
            "Parameters shown for MetricGAN+: the enhancement model itself is fixed "
            "(speechbrain/metricgan-plus-voicebank). The app converts input to mono 16 kHz "
            "for the model, restores the original sample rate afterward, and uses 30 second "
            "chunks with 0.5 seconds overlap on long files. Output level controls still apply.",
            font_size=10,
            panel=True,
        ))

        self._metricgan_notice = QLabel("")
        self._metricgan_notice.setWordWrap(True)
        self._metricgan_notice.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 2px;")
        self._metricgan_notice.setVisible(False)
        mg_layout.addWidget(self._metricgan_notice)

        self._metricgan_widget.setVisible(False)
        enhance_layout.addWidget(self._metricgan_widget)

        # Connect engine selector
        self._enhance_engine_combo.currentIndexChanged.connect(self._on_enhance_engine_changed)

        # ── Enhancement tab ──
        enh_tab = QWidget()
        enh_tab_layout = QVBoxLayout(enh_tab)
        enh_tab_layout.setContentsMargins(4, 4, 4, 4)
        enh_tab_layout.setSpacing(6)
        self._main_tabs.addTab(enh_tab, "Enhancement")

        # Move enhance_options content directly into the enhancement tab
        self._enhance_options.setVisible(True)
        enh_tab_layout.addWidget(self._enhance_options)
        enh_tab_layout.addStretch()

        # Connect main tab changes
        self._main_tabs.currentChanged.connect(self._on_main_tab_changed)

        # Output format
        fmt_group = QGroupBox("Output Format")
        fmt_layout = QHBoxLayout(fmt_group)
        self._fmt_group = QButtonGroup(self)
        for fmt in ["WAV", "FLAC", "MP3"]:
            rb = QRadioButton(fmt)
            self._fmt_group.addButton(rb)
            fmt_layout.addWidget(rb)
            if fmt == "WAV":
                rb.setChecked(True)
        layout.addWidget(fmt_group)

        # Output directory
        out_group = QGroupBox("Output Directory")
        out_layout = QVBoxLayout(out_group)
        self._output_label = QLabel("Default: subfolder named after input file")
        self._output_label.setWordWrap(True)
        self._output_label.setStyleSheet("color: #a6adc8; padding: 4px;")
        out_btn = QPushButton("Change Output Directory...")
        out_btn.clicked.connect(self._set_output_dir)
        out_layout.addWidget(self._output_label)
        out_layout.addWidget(out_btn)
        layout.addWidget(out_group)

        # Validate Settings button (SAM only, shown before Isolate Sound)
        layout.addSpacing(4)
        self._validate_btn = QPushButton("Validate Settings")
        self._validate_btn.setMinimumHeight(40)
        self._validate_btn.setStyleSheet(
            "QPushButton { background-color: #f9e2af; color: #1e1e2e; font-weight: bold; "
            "border: 1px solid #f9e2af; border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #f5c97e; }"
            "QPushButton:disabled { background-color: #45475a; color: #585b70; }")
        self._validate_btn.clicked.connect(self._validate_sam_settings)
        self._validate_btn.setEnabled(False)
        self._validate_btn.setVisible(False)
        layout.addWidget(self._validate_btn)

        # Separate / Isolate button
        self._separate_btn = QPushButton("Separate Audio")
        self._separate_btn.setObjectName("primaryBtn")
        self._separate_btn.setMinimumHeight(40)
        self._separate_btn.clicked.connect(self._start_separation)
        self._separate_btn.setEnabled(False)
        layout.addWidget(self._separate_btn)

        self._stop_btn = QPushButton("Stop Processing")
        self._stop_btn.setObjectName("dangerBtn")
        self._stop_btn.clicked.connect(self._stop_separation)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setVisible(False)
        layout.addWidget(self._stop_btn)

        # Global Effects (hidden until separation is done)
        self._fx_group = QGroupBox("Audio Effects")
        fx_layout = QVBoxLayout(self._fx_group)
        self._global_fx_btn = QPushButton("Open Effects Panel...")
        self._global_fx_btn.clicked.connect(self._open_global_effects)
        fx_layout.addWidget(self._global_fx_btn)
        self._fx_group.setVisible(False)
        layout.addWidget(self._fx_group)

        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(4)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #a6adc8;")
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_label)

        # Header: filename + KEY + BPM
        header = QHBoxLayout()
        self._file_title = QLabel("")
        self._file_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #cdd6f4;")
        header.addWidget(self._file_title)
        header.addStretch()
        self._key_label = QLabel("")
        self._key_label.setStyleSheet("color: #a6adc8; font-size: 13px;")
        header.addWidget(self._key_label)
        header.addSpacing(20)
        self._bpm_label = QLabel("")
        self._bpm_label.setStyleSheet("color: #a6adc8; font-size: 13px;")
        header.addWidget(self._bpm_label)
        layout.addLayout(header)

        # Initial waveform (shown before separation, hidden after)
        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._on_waveform_seek)
        layout.addWidget(self._waveform)

        # Scrollable stem tracks area
        self._stems_scroll = QScrollArea()
        self._stems_scroll.setWidgetResizable(True)
        self._stems_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._stems_scroll.setVisible(False)
        self._stems_container = QWidget()
        self._stems_layout = QVBoxLayout(self._stems_container)
        self._stems_layout.setSpacing(4)
        self._stems_layout.setContentsMargins(0, 0, 0, 0)
        self._stems_layout.addStretch()
        self._stems_scroll.setWidget(self._stems_container)
        layout.addWidget(self._stems_scroll, stretch=1)

        # Transport bar
        self._transport = TransportBar()
        self._transport.play_clicked.connect(self._play_all)
        self._transport.pause_clicked.connect(self._engine.pause)
        self._transport.stop_clicked.connect(self._engine.stop)
        self._transport.split_another_clicked.connect(self._reset_for_new_file)
        layout.addWidget(self._transport)

        return panel

    def _build_status_bar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._gpu_label = QPushButton("GPU: Checking...")
        self._gpu_label.setFlat(True)
        self._gpu_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gpu_label.setStyleSheet(
            "QPushButton { border: none; background: transparent; padding: 0 6px; font-size: 11px; }"
        )
        self._gpu_label.clicked.connect(lambda: self._open_settings("acceleration"))
        self._status_bar.addPermanentWidget(self._gpu_label)
        self._status_bar.showMessage("Ready")

    # ── Log Panel ──

    def _toggle_log_panel(self, checked: bool):
        self._log_viewer.setVisible(checked)
        total_height = self._vsplitter.size().height()
        if total_height <= 0:
            total_height = max(self.height(), self.minimumHeight())
        if checked:
            top_height = max(420, int(total_height * 0.72))
            self._vsplitter.setSizes([top_height, max(160, total_height - top_height)])
        else:
            self._vsplitter.setSizes([total_height, 0])

    # ── GPU Detection ──

    def _update_gpu_info(self):
        report = probe_acceleration(self._settings)
        self._acceleration_report = report
        style = (
            "QPushButton { border: none; background: transparent; padding: 0 6px; "
            "font-size: 11px; color: %s; }"
        )
        if report.mode == "gpu":
            text = f"GPU: {report.gpu_name or 'Ready'}"
            color = "#a6e3a1"
        elif report.reason_code in {"ffmpeg_missing", "no_nvidia_gpu"}:
            text = "GPU: CPU mode"
            color = "#f9e2af"
        else:
            text = "GPU: Setup needed"
            color = "#f38ba8"
        self._gpu_label.setText(text)
        self._gpu_label.setToolTip(report.summary + "\n\n" + report.details + "\n\nClick to open Settings.")
        self._gpu_label.setStyleSheet(style % color)

    def _open_settings(self, initial_tab: str = "general"):
        dialog = SettingsDialog(self._settings, initial_tab=initial_tab, parent=self)
        if dialog.exec():
            self._settings = get_app_settings()
            self._settings.apply_runtime_environment()
            self._refresh_output_label()
            self._update_gpu_info()
            self._status_bar.showMessage("Settings saved.")

    def _refresh_output_label(self):
        if self._output_dir:
            text = f"Session override: {self._output_dir}"
            color = "#a6e3a1"
        elif self._settings.output_mode() == "fixed" and self._settings.default_output_dir():
            text = f"Default output folder: {self._settings.default_output_dir()}"
            color = "#89b4fa"
        else:
            text = "Default: subfolder named after input file"
            color = "#a6adc8"
        self._output_label.setText(text)
        self._output_label.setStyleSheet(f"color: {color}; padding: 4px;")

    def _ensure_screen_tracking(self) -> None:
        if self._screen_change_connected:
            return
        handle = self.windowHandle()
        if handle is None:
            return
        handle.screenChanged.connect(self._on_screen_changed)
        self._screen_change_connected = True

    def _current_screen_geometry(self):
        self._ensure_screen_tracking()
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else None
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        return screen.availableGeometry()

    def _target_window_size_for_screen(self) -> tuple[int, int] | None:
        geometry = self._current_screen_geometry()
        if geometry is None:
            return None
        max_width = max(900, geometry.width() - 24)
        max_height = max(640, geometry.height() - 24)
        target_width = min(max_width, 1400)
        if target_width < 1024:
            target_width = max(960, max_width)
        target_height = min(max_height, max(780, int(geometry.height() * 0.88)))
        return target_width, target_height

    def _apply_adaptive_window_geometry(self, *, force: bool = False, center: bool = False) -> None:
        geometry = self._current_screen_geometry()
        target_size = self._target_window_size_for_screen()
        if geometry is None or target_size is None:
            return

        current_size = self.size()
        should_resize = force or (
            current_size.width() > geometry.width()
            or current_size.height() > geometry.height()
            or current_size.width() < self.minimumWidth()
            or current_size.height() < self.minimumHeight()
        )
        if should_resize:
            self.resize(*target_size)

        if center:
            frame = self.frameGeometry()
            frame.moveCenter(geometry.center())
            self.move(frame.topLeft())
        else:
            frame = self.frameGeometry()
            if (
                frame.right() > geometry.right()
                or frame.bottom() > geometry.bottom()
                or frame.left() < geometry.left()
                or frame.top() < geometry.top()
            ):
                frame.moveCenter(geometry.center())
                self.move(frame.topLeft())

    def _apply_splitter_defaults(self) -> None:
        if self._hsplitter is None:
            return
        total_width = self._hsplitter.size().width()
        if total_width <= 0:
            total_width = max(self.width(), self.minimumWidth())
        left_width = max(320, min(420, int(total_width * 0.28)))
        right_width = max(560, total_width - left_width)
        self._hsplitter.setSizes([left_width, right_width])

    def _on_screen_changed(self, *_args) -> None:
        self._apply_adaptive_window_geometry(force=False, center=False)
        QTimer.singleShot(0, self._apply_splitter_defaults)

    def _dialog_start_dir(self, *, output: bool) -> str:
        if output:
            if self._output_dir:
                return self._output_dir
            last_output = self._settings.last_output_dir()
            if self._settings.reuse_last_used_folder() and last_output:
                return last_output
            default_output = self._settings.default_output_dir()
            if default_output:
                return default_output
            if self._input_file:
                return str(Path(self._input_file).parent)
        else:
            if self._original_input_path:
                return str(Path(self._original_input_path).parent)
            last_input = self._settings.last_input_dir()
            if self._settings.reuse_last_used_folder() and last_input:
                return last_input
            default_output = self._settings.default_output_dir()
            if default_output:
                return default_output
        return str(Path.home())

    def _resolve_output_dir(self, input_path: str | None = None, *, batch: bool = False) -> str:
        if self._output_dir:
            return self._output_dir
        default_output = self._settings.default_output_dir()
        if self._settings.output_mode() == "fixed" and default_output:
            return default_output
        if not input_path:
            return default_output or str(Path.home())
        source = Path(input_path)
        if batch:
            return str(source.parent / "separated")
        return str(source.parent / source.stem)

    def _refresh_tools_state(self):
        if self._split_clip_action is not None:
            self._split_clip_action.setEnabled(bool(self._input_file and self._original_input_path))
        if self._extract_audio_action is not None:
            is_loaded_video = bool(
                self._original_input_path
                and Path(self._original_input_path).suffix.lower() in self._VIDEO_EXTS
            )
            self._extract_audio_action.setEnabled(is_loaded_video)

    # ── File Operations ──

    def _clear_effects_session(self):
        self._effects_preview_session += 1
        self._effects_preview_latest_requests.clear()
        self._effects_preview_processor.clear_pending()
        self._layer_effect_states.clear()
        self._preview_layer_effect_states.clear()
        self._layer_labels.clear()
        if self._effects_dialog is not None:
            self._suppress_effect_dialog_discard = True
            self._effects_dialog.close()
            self._suppress_effect_dialog_discard = False
            self._effects_dialog = None

    _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma", ".aac"}
    _VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

    def _media_file_filter(self) -> str:
        return (
            "Audio/Video Files (*.wav *.mp3 *.flac *.m4a *.ogg *.wma *.aac "
            "*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*)"
        )

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio/Video File", self._dialog_start_dir(output=False),
            self._media_file_filter(),
        )
        if path:
            self._set_input_file(path)

    def _set_input_file(self, path: str):
        self._clear_effects_session()
        # Track original path for visual prompting (before video extraction)
        self._original_input_path = path
        self._settings.set_last_input_dir(str(Path(path).parent))
        self._settings.sync()
        is_video = Path(path).suffix.lower() in self._VIDEO_EXTS

        # Extract audio from video files
        if is_video:
            extracted = self._extract_audio_from_video(path)
            if extracted is None:
                return
            path = extracted

        self._engine.stop()
        self._input_file = path
        self._refresh_tools_state()

        # Update visual prompt button (video files only)
        self._visual_prompt_btn.setEnabled(is_video)
        self._clear_visual_mask()
        name = Path(path).name
        self._input_label.setText(name)
        self._input_label.setStyleSheet("color: #a6e3a1; padding: 4px;")
        self._file_title.setText(name)
        is_sam = (self._main_tabs.currentIndex() == 0 and self._sub_cat_combo.currentIndex() == 1)
        if is_sam:
            self._validate_btn.setEnabled(True)
            self._separate_btn.setEnabled(False)
            self._separate_btn.setVisible(False)
        else:
            self._separate_btn.setEnabled(True)
        self._status_bar.showMessage(f"Loaded: {name}")

        # Reset to initial view
        self._waveform.setVisible(True)
        self._stems_scroll.setVisible(False)
        self._clear_stem_tracks()

        # Reset analysis labels
        self._key_label.setText("")
        self._bpm_label.setText("")

        # Load waveform
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            display = data.mean(axis=1) if data.ndim == 2 else data
            self._waveform.plot_waveform(display, sr, title=name)
            self._engine.load_original(path)
            self._transport.set_duration(len(data) / sr)
        except Exception as e:
            self._status_bar.showMessage(f"Could not read waveform: {e}")

        # Start BPM/key analysis
        self._analysis_worker = AudioAnalysisWorker(path, self)
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(lambda e: self._status_bar.showMessage(f"Analysis: {e}"))
        self._analysis_worker.start()

    def _extract_audio_from_video(self, video_path: str) -> str | None:
        """Extract audio from a video file using ffmpeg. Returns path to WAV or None."""
        output_dir = self._resolve_output_dir(video_path)
        try:
            self._status_bar.showMessage(f"Extracting audio from {Path(video_path).name}...")
            wav_path = extract_audio_from_video(video_path, output_dir)
            self._status_bar.showMessage(f"Audio extracted: {Path(wav_path).name}")
            return wav_path
        except FileNotFoundError:
            QMessageBox.critical(self, "FFmpeg Not Found",
                "FFmpeg is required to extract audio from video files.\n\n"
                "Install FFmpeg or use the bundled app build.")
            return None
        except Exception as e:
            QMessageBox.critical(self, "Extraction Failed",
                f"Could not extract audio from video:\n\n{str(e)[:300]}")
            self._status_bar.showMessage("Audio extraction failed.")
            return None

    def _on_analysis_done(self, bpm: float, key: str):
        self._key_label.setText(f"KEY    {key}")
        self._key_label.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")
        self._bpm_label.setText(f"BPM    {int(bpm)}")
        self._bpm_label.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")

    def _open_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Audio Files",
            self._dialog_start_dir(output=False),
        )
        if not folder:
            return
        self._settings.set_last_input_dir(folder)
        self._settings.sync()
        exts = self._AUDIO_EXTS | self._VIDEO_EXTS
        files = [str(p) for p in Path(folder).iterdir() if p.suffix.lower() in exts]
        if not files:
            QMessageBox.information(self, "No Audio/Video Files", "No supported files found.")
            return
        self._start_batch_separation(files)

    def _set_output_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._dialog_start_dir(output=True),
        )
        if folder:
            self._output_dir = folder
            self._settings.set_last_output_dir(folder)
            self._settings.sync()
            self._refresh_output_label()

    def _open_split_dialog(self):
        source_path = self._original_input_path or self._input_file
        waveform_source = self._input_file or source_path
        if not source_path or not waveform_source:
            QMessageBox.information(self, "No File Loaded", "Load an audio or video file first.")
            return
        if not Path(source_path).exists():
            QMessageBox.warning(self, "File Missing", f"The source file could not be found:\n\n{source_path}")
            return
        if not Path(waveform_source).exists():
            QMessageBox.warning(
                self,
                "Waveform Missing",
                f"The waveform source file could not be found:\n\n{waveform_source}",
            )
            return

        self._show_split_dialog(source_path, waveform_source)

    def _open_split_dialog_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Audio/Video File to Split",
            self._dialog_start_dir(output=False),
            self._media_file_filter(),
        )
        if not path:
            return
        prepared = self._prepare_split_dialog_inputs(path)
        if prepared is None:
            return
        source_path, waveform_source_path, cleanup_dir = prepared
        self._show_split_dialog(source_path, waveform_source_path, cleanup_dir=cleanup_dir)

    def _prepare_split_dialog_inputs(self, source_path: str) -> tuple[str, str, str | None] | None:
        source = Path(source_path)
        if not source.exists():
            QMessageBox.warning(self, "File Missing", f"The selected file could not be found:\n\n{source_path}")
            return None

        self._settings.set_last_input_dir(str(source.parent))
        self._settings.sync()

        if source.suffix.lower() not in self._VIDEO_EXTS:
            return source_path, source_path, None

        temp_dir = tempfile.mkdtemp(prefix="ai-audio-toolkit-split-")
        try:
            self._status_bar.showMessage(f"Preparing waveform for {source.name}...")
            waveform_source = extract_audio_from_video(source_path, temp_dir)
            return source_path, waveform_source, temp_dir
        except FileNotFoundError:
            shutil.rmtree(temp_dir, ignore_errors=True)
            QMessageBox.critical(
                self,
                "FFmpeg Not Found",
                "FFmpeg is required to split video files.\n\n"
                "Install FFmpeg, configure it in Settings, or use the bundled app build.",
            )
            return None
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            QMessageBox.critical(
                self,
                "Preparation Failed",
                f"Could not prepare the selected video for splitting:\n\n{str(exc)[:300]}",
            )
            self._status_bar.showMessage("Split preparation failed.")
            return None

    def _show_split_dialog(self, source_path: str, waveform_source_path: str, *, cleanup_dir: str | None = None):
        try:
            dialog = SplitClipDialog(
                source_path=source_path,
                waveform_source_path=waveform_source_path,
                output_dir=self._resolve_output_dir(source_path),
                parent=self,
            )
            if dialog.exec():
                self._status_bar.showMessage("Clip exported.")
            else:
                self._status_bar.showMessage("Split clip closed.")
        finally:
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def _extract_audio_for_loaded_video(self):
        video_path = self._original_input_path
        if not video_path or Path(video_path).suffix.lower() not in self._VIDEO_EXTS:
            QMessageBox.information(self, "No Video Loaded", "Load a video file first.")
            return
        self._run_extract_audio_tool(video_path)

    def _extract_audio_from_video_file(self):
        video_filter = "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*)"
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Video File to Extract Audio From",
            self._dialog_start_dir(output=False),
            video_filter,
        )
        if not video_path:
            return
        self._settings.set_last_input_dir(str(Path(video_path).parent))
        self._settings.sync()
        self._run_extract_audio_tool(video_path)

    def _run_extract_audio_tool(self, video_path: str):
        source = Path(video_path)
        if not source.exists():
            QMessageBox.warning(self, "File Missing", f"The selected video could not be found:\n\n{video_path}")
            return
        if source.suffix.lower() not in self._VIDEO_EXTS:
            QMessageBox.warning(self, "Unsupported File", "Choose a supported video file.")
            return

        default_output = Path(self._resolve_output_dir(video_path)) / f"{source.stem}_extracted.wav"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Extracted Audio",
            str(default_output),
            "WAV Audio (*.wav)",
        )
        if not save_path:
            return
        if Path(save_path).suffix.lower() != ".wav":
            save_path = f"{save_path}.wav"

        try:
            self._status_bar.showMessage(f"Extracting audio from {source.name}...")
            extracted_path = extract_audio_from_video_to_path(video_path, save_path)
            self._settings.set_last_output_dir(str(Path(extracted_path).parent))
            self._settings.sync()
            self._status_bar.showMessage(f"Audio extracted: {Path(extracted_path).name}")
            QMessageBox.information(
                self,
                "Audio Extracted",
                f"Saved extracted audio to:\n\n{extracted_path}",
            )
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "FFmpeg Not Found",
                "FFmpeg is required to extract audio from video files.\n\n"
                "Install FFmpeg, configure it in Settings, or use the bundled app build.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Extraction Failed",
                f"Could not extract audio from video:\n\n{str(exc)[:300]}",
            )
            self._status_bar.showMessage("Audio extraction failed.")

    def _reset_for_new_file(self):
        self._engine.stop()
        self._clear_effects_session()
        self._clear_stem_tracks()
        self._waveform.setVisible(True)
        self._stems_scroll.setVisible(False)
        self._waveform.clear()
        self._file_title.setText("")
        self._key_label.setText("")
        self._bpm_label.setText("")
        self._input_file = None
        self._original_input_path = None
        self._output_files.clear()
        self._separate_btn.setEnabled(False)
        self._validate_btn.setEnabled(False)
        self._chunk_group.setVisible(False)
        self._clear_visual_mask()
        self._fx_group.setVisible(False)
        self._refresh_tools_state()
        self._input_label.setText("No file selected (drag & drop or click Open)")
        self._input_label.setStyleSheet("color: #a6adc8; padding: 4px;")
        self._transport.set_duration(0)
        self._refresh_output_label()
        self._status_bar.showMessage("Ready")

    # ── Tab / Category Selection ──

    def _on_main_tab_changed(self, tab_idx: int):
        """Switch between Separation and Enhancement tabs."""
        if tab_idx == 1:
            # Enhancement tab
            self._separate_btn.setText("Enhance Audio")
            self._separate_btn.setVisible(True)
            self._separate_btn.setEnabled(self._input_file is not None)
            self._validate_btn.setVisible(False)
            self._refresh_enhance_notice(self._enhance_engine_combo.currentIndex())
        else:
            # Separation tab — restore state based on sub-category
            self._on_sep_subcategory_changed(self._sub_cat_combo.currentIndex())

    def _on_sep_subcategory_changed(self, idx: int):
        """Toggle between Standard Models and SAM-Audio within the Separation tab."""
        is_sam = idx == 1
        is_standard = not is_sam

        self._model_tree.setVisible(is_standard)
        self._model_desc.setVisible(is_standard)
        self._model_params_group.setVisible(False)
        self._sam_options.setVisible(is_sam)

        if is_sam:
            self._separate_btn.setText("Isolate Sound")
            self._separate_btn.setVisible(False)  # hidden until validated
            self._validate_btn.setVisible(True)
            self._validate_btn.setEnabled(self._input_file is not None)
            self._refresh_sam_notice()
            is_video = (self._original_input_path is not None
                        and Path(self._original_input_path).suffix.lower() in self._VIDEO_EXTS)
            self._visual_prompt_btn.setEnabled(is_video)
            self._model_tree.clear()
        else:
            self._separate_btn.setText("Separate Audio")
            self._separate_btn.setVisible(True)
            self._validate_btn.setVisible(False)
            # Populate with first sub-category presets or keep current selection
            if self._model_tree.topLevelItemCount() == 0:
                first_cat = next(iter(MODEL_PRESETS), None)
                if first_cat:
                    self._populate_model_tree(first_cat)

    def _populate_model_tree(self, category: str):
        self._model_tree.clear()
        presets = MODEL_PRESETS.get(category, [])
        for preset in presets:
            item = QTreeWidgetItem([f"{preset.name}  [{preset.architecture}]"])
            item.setData(0, Qt.ItemDataRole.UserRole, preset.name)
            self._model_tree.addTopLevelItem(item)
        if self._model_tree.topLevelItemCount() > 0:
            self._model_tree.setCurrentItem(self._model_tree.topLevelItem(0))

    # Keep for compatibility — called from _on_model_file_opened and similar
    def _on_category_changed(self, category: str):
        """Legacy dispatcher — maps category string to the new tab/sub-category system."""
        if category == "Audio Enhancement":
            self._main_tabs.setCurrentIndex(1)
        elif category == "SAM-Audio":
            self._main_tabs.setCurrentIndex(0)
            self._sub_cat_combo.setCurrentIndex(1)
        else:
            self._main_tabs.setCurrentIndex(0)
            self._sub_cat_combo.setCurrentIndex(0)
            self._populate_model_tree(category)

    def _refresh_sam_notice(self):
        if check_sam_available():
            self._sam_notice.setVisible(False)
            self._sam_notice.setText("")
            return

        from .sam_backend import SAM_IMPORT_ERROR

        message = "sam-audio not installed.\nInstall: pip install sam-audio"
        if SAM_IMPORT_ERROR:
            message = f"sam-audio import error:\n{SAM_IMPORT_ERROR[:120]}"
        self._sam_notice.setText(message)
        self._sam_notice.setVisible(True)

    @staticmethod
    def _set_notice_text(label: QLabel, message: str | None) -> None:
        text = (message or "").strip()
        label.setText(text)
        label.setVisible(bool(text))

    def _backend_notice_message(
        self,
        kind: str,
        *,
        missing_message: str,
        import_error_prefix: str,
    ) -> str | None:
        probe = get_backend_probe_state(kind)
        if probe["checked"]:
            if probe["available"]:
                return None
            error = str(probe["error"] or "").strip()
            if error:
                return f"{import_error_prefix}\n{error[:120]}"
            return missing_message
        if not probe["installed"]:
            return missing_message
        return None

    def _refresh_studio_sound_notice(self) -> None:
        message = self._backend_notice_message(
            "deepfilter",
            missing_message=(
                "Note: DeepFilterNet is not installed, so the noise-removal stage will be skipped.\n"
                "Install: pip install deepfilternet"
            ),
            import_error_prefix="DeepFilterNet import error:",
        )
        self._set_notice_text(self._studio_sound_notice, message)

    def _refresh_enhance_notice(self, index: int):
        self._refresh_studio_sound_notice()

        if index == 1:
            self._set_notice_text(
                self._deepfilter_notice,
                self._backend_notice_message(
                    "deepfilter",
                    missing_message="deepfilternet not installed.\nInstall: pip install deepfilternet",
                    import_error_prefix="deepfilternet import error:",
                ),
            )
            return

        if index == 3:
            self._set_notice_text(
                self._clearvoice_notice,
                self._backend_notice_message(
                    "clearvoice",
                    missing_message=(
                        "clearvoice not installed.\n"
                        "Install manually: pip install clearvoice --no-deps\n"
                        "Note: --no-deps avoids downgrading shared packages in this app environment."
                    ),
                    import_error_prefix="clearvoice import error:",
                ),
            )
            return

        if index == 4:
            self._set_notice_text(
                self._mdxnet_notice,
                self._backend_notice_message(
                    "audio_separator",
                    missing_message=(
                        "audio-separator is not installed.\n"
                        "Install: pip install \"audio-separator[gpu]>=0.25.0\""
                    ),
                    import_error_prefix="audio-separator import error:",
                ),
            )
            return

        if index == 5:
            self._set_notice_text(
                self._voicefixer_notice,
                self._backend_notice_message(
                    "voicefixer",
                    missing_message="voicefixer not installed — will auto-install when you press Enhance Audio.",
                    import_error_prefix="voicefixer import error:",
                ),
            )
            return

        if index == 6:
            self._set_notice_text(
                self._metricgan_notice,
                self._backend_notice_message(
                    "metricgan",
                    missing_message="speechbrain not installed — will auto-install when you press Enhance Audio.",
                    import_error_prefix="speechbrain import error:",
                ),
            )
            return

        self._set_notice_text(
            self._resemble_notice,
            self._backend_notice_message(
                "resemble",
                missing_message=(
                    "resemble-enhance not installed.\n"
                    "Install: pip install resemble-enhance --no-deps"
                ),
                import_error_prefix="resemble-enhance import error:",
            ),
        )

    def _on_enhance_engine_changed(self, index: int):
        """Toggle between enhancement engine controls."""
        self._resemble_widget.setVisible(index == 0)
        self._deepfilter_widget.setVisible(index == 1)
        self._studio_sound_widget.setVisible(index == 2)
        self._clearvoice_widget.setVisible(index == 3)
        self._mdxnet_widget.setVisible(index == 4)
        self._voicefixer_widget.setVisible(index == 5)
        self._metricgan_widget.setVisible(index == 6)
        self._refresh_enhance_notice(index)

    def _update_enhance_level_hint(self, *_args):
        if not hasattr(self, "_enhance_level_hint"):
            return

        auto_level = self._enhance_auto_level_cb.isChecked()
        match_level = self._enhance_match_level_cb.isChecked()
        gain_db = self._enhance_gain_slider.value()

        lines = [
            "Order: 1. match input loudness, 2. auto level active speech and trim peak, "
            "3. output gain, 4. safety limiter."
        ]
        if auto_level and match_level:
            lines.append(
                "Both first steps are active: they stack. Match input gets the file near "
                "the source loudness first; auto level can then move it toward a consistent "
                "speech level, so it may change the matched loudness."
            )
        elif match_level:
            lines.append("Only match input is active: the saved file stays close to the source loudness.")
        elif auto_level:
            lines.append("Only auto level is active: the saved file targets a consistent final speech level.")
        else:
            lines.append("Both automatic levelers are off: only the output gain and limiter can change level.")

        if gain_db:
            lines.append(
                f"Output gain is a final {gain_db:+d} dB trim after the automatic steps; "
                "the limiter may reduce it if it would clip."
            )
        else:
            lines.append("Output gain is at 0 dB, so it is not changing the leveled result.")

        self._enhance_level_hint.setText("\n".join(lines))

    def _on_enhance_mode_changed(self, index: int):
        """Toggle CFM settings visibility based on enhance/denoise mode."""
        is_full_enhance = index == 0  # "Enhance" vs "Denoise only"
        self._cfm_group.setVisible(is_full_enhance)
        self._denoise_before_cb.setVisible(is_full_enhance)

    def _update_model_params_visibility(self, arch: str):
        """Show/hide model parameter controls based on architecture."""
        self._model_params_group.setVisible(True)

        is_mdx = arch == "MDX-Net"
        is_vr = arch == "VR"
        is_demucs = arch == "Demucs"
        is_mdxc = arch in ("MDXC", "Roformer")

        # Row visibility by architecture
        self._param_segment_row.setVisible(is_mdx or is_demucs or is_mdxc)
        self._param_overlap_row.setVisible(is_mdx or is_demucs)
        self._param_overlap_int_row.setVisible(is_mdxc)
        self._param_batch_row.setVisible(is_mdx or is_vr or is_mdxc)
        self._param_shifts_row.setVisible(is_demucs)
        self._param_window_row.setVisible(is_vr)
        self._param_aggression_row.setVisible(is_vr)
        self._param_pitch_row.setVisible(is_mdxc)

        # Checkboxes
        self._param_denoise_cb.setVisible(is_mdx)
        self._param_tta_cb.setVisible(is_vr)
        self._param_high_end_cb.setVisible(is_vr)
        self._param_post_process_cb.setVisible(is_vr)
        self._param_segments_enabled_cb.setVisible(is_demucs)

        # Architecture-specific help text
        help_texts = {
            "MDX-Net": (
                "MDX-Net tips:\n"
                "• Clean studio audio: Segment 256, Overlap 0.25 — fast and accurate.\n"
                "• Noisy recordings: enable Denoise pass + Overlap 0.50.\n"
                "• VRAM tight (4–6GB)? Segment 128, Batch 1.\n"
                "• Batch size speeds up processing only — zero effect on quality.\n"
                "• Kim Vocal 2 is the fastest MDX-Net model; good for quick previews."),
            "VR": (
                "VR architecture tips:\n"
                "• Vocals: Window 320 gives the clearest detail; use 512 for speed.\n"
                "• Instruments (drums, bass): Aggression 4 avoids muddiness.\n"
                "• TTA: +15–20% quality, 3–4× slower — save for final exports.\n"
                "• Post-process: last resort only for stubborn vocal bleed in instrumental.\n"
                "• High-end processing: try if output sounds dull or lacks 'air'."),
            "Demucs": (
                "HTDemucs tips:\n"
                "• Shifts 2 is the sweet spot for quality vs speed.\n"
                "• Live/concert recordings: Shifts 5 + Overlap 0.50.\n"
                "• GPU strongly recommended for Shifts > 0; CPU = very slow.\n"
                "• Fine-Tuned 4-stem variant outperforms Standard for every stem.\n"
                "• Disable segments only on 24GB+ VRAM workstation GPUs."),
            "MDXC": (
                "MDXC tips:\n"
                "• Overlap 8 default; increase to 12–16 to reduce boundary artifacts.\n"
                "• Pitch shift +2: helps on bass-heavy mixes (kick, 808, bass guitar).\n"
                "• Pitch shift -2: helps on bright/treble-heavy mixes (violins, cymbals).\n"
                "• Segment 256 for 8GB VRAM; reduce to 128 if out-of-memory errors."),
            "Roformer": (
                "Roformer tips:\n"
                "• Highest quality architecture — use for final exports.\n"
                "• BS-Roformer: best overall vocal separation (SDR 12.97).\n"
                "• MelBand Roformer: slightly faster; excellent for karaoke tracks.\n"
                "• De-Echo/DeReverb: Overlap 12–16 for cleanest room removal.\n"
                "• Pitch shift ±2 can help if separation seems incomplete on extreme audio."),
        }
        help_text = help_texts.get(arch, "")
        self._param_help.setText(help_text)
        self._param_help.setVisible(bool(help_text))

    def _on_param_preset_changed(self, preset_name: str):
        """Apply parameter preset values."""
        preset = self._get_selected_preset()
        if not preset:
            return
        arch = preset.architecture

        if preset_name == "Default":
            values = self._get_default_params(arch)
        elif preset_name == "High Quality":
            values = self._get_hq_params(arch)
        elif preset_name == "Fast":
            values = self._get_fast_params(arch)
        elif preset_name == "Low VRAM":
            values = self._get_low_vram_params(arch)
        else:
            return

        # Apply values to sliders/combos
        if "segment_size" in values:
            self._param_segment_slider.setValue(values["segment_size"])
        if "overlap" in values:
            self._param_overlap_slider.setValue(int(values["overlap"] * 100))
        if "overlap_int" in values:
            self._param_overlap_int_slider.setValue(values["overlap_int"])
        if "batch_size" in values:
            self._param_batch_slider.setValue(values["batch_size"])
        if "shifts" in values:
            self._param_shifts_slider.setValue(values["shifts"])
        if "window_size" in values:
            idx = {320: 0, 512: 1, 1024: 2}.get(values["window_size"], 1)
            self._param_window_combo.setCurrentIndex(idx)
        if "aggression" in values:
            self._param_aggression_slider.setValue(values["aggression"])
        if "pitch_shift" in values:
            self._param_pitch_slider.setValue(values["pitch_shift"])

    @staticmethod
    def _get_default_params(arch: str) -> dict:
        if arch == "MDX-Net":
            return {"segment_size": 256, "overlap": 0.25, "batch_size": 1}
        if arch == "VR":
            return {"window_size": 512, "aggression": 5, "batch_size": 1}
        if arch == "Demucs":
            return {"segment_size": 256, "overlap": 0.25, "shifts": 2}
        if arch in ("MDXC", "Roformer"):
            return {"segment_size": 256, "overlap_int": 8, "batch_size": 1, "pitch_shift": 0}
        return {}

    @staticmethod
    def _get_hq_params(arch: str) -> dict:
        if arch == "MDX-Net":
            return {"segment_size": 512, "overlap": 0.50, "batch_size": 1}
        if arch == "VR":
            return {"window_size": 320, "aggression": 5, "batch_size": 1}
        if arch == "Demucs":
            return {"segment_size": 256, "overlap": 0.50, "shifts": 5}
        if arch in ("MDXC", "Roformer"):
            return {"segment_size": 256, "overlap_int": 16, "batch_size": 1, "pitch_shift": 0}
        return {}

    @staticmethod
    def _get_fast_params(arch: str) -> dict:
        if arch == "MDX-Net":
            return {"segment_size": 128, "overlap": 0.10, "batch_size": 4}
        if arch == "VR":
            return {"window_size": 1024, "aggression": 5, "batch_size": 4}
        if arch == "Demucs":
            return {"segment_size": 256, "overlap": 0.10, "shifts": 0}
        if arch in ("MDXC", "Roformer"):
            return {"segment_size": 128, "overlap_int": 4, "batch_size": 4, "pitch_shift": 0}
        return {}

    @staticmethod
    def _get_low_vram_params(arch: str) -> dict:
        if arch == "MDX-Net":
            return {"segment_size": 64, "overlap": 0.25, "batch_size": 1}
        if arch == "VR":
            return {"window_size": 512, "aggression": 5, "batch_size": 1}
        if arch == "Demucs":
            return {"segment_size": 64, "overlap": 0.25, "shifts": 1}
        if arch in ("MDXC", "Roformer"):
            return {"segment_size": 64, "overlap_int": 8, "batch_size": 1, "pitch_shift": 0}
        return {}

    def _build_separator_params(self) -> dict:
        """Build architecture-specific parameter dict for the Separator."""
        preset = self._get_selected_preset()
        if not preset:
            return {}
        arch = preset.architecture

        if arch == "MDX-Net":
            return {
                "mdx_params": {
                    "hop_length": 1024,
                    "segment_size": self._param_segment_slider.value(),
                    "overlap": self._param_overlap_slider.value() / 100.0,
                    "batch_size": self._param_batch_slider.value(),
                    "enable_denoise": self._param_denoise_cb.isChecked(),
                },
            }
        if arch == "VR":
            ws_map = {0: 320, 1: 512, 2: 1024}
            return {
                "vr_params": {
                    "batch_size": self._param_batch_slider.value(),
                    "window_size": ws_map.get(self._param_window_combo.currentIndex(), 512),
                    "aggression": self._param_aggression_slider.value(),
                    "enable_tta": self._param_tta_cb.isChecked(),
                    "enable_post_process": self._param_post_process_cb.isChecked(),
                    "post_process_threshold": 0.2,
                    "high_end_process": self._param_high_end_cb.isChecked(),
                },
            }
        if arch == "Demucs":
            seg = self._param_segment_slider.value()
            return {
                "demucs_params": {
                    "segment_size": seg if seg != 256 else "Default",
                    "shifts": self._param_shifts_slider.value(),
                    "overlap": self._param_overlap_slider.value() / 100.0,
                    "segments_enabled": self._param_segments_enabled_cb.isChecked(),
                },
            }
        if arch in ("MDXC", "Roformer"):
            return {
                "mdxc_params": {
                    "segment_size": self._param_segment_slider.value(),
                    "override_model_segment_size": False,
                    "batch_size": self._param_batch_slider.value(),
                    "overlap": self._param_overlap_int_slider.value(),
                    "pitch_shift": self._param_pitch_slider.value(),
                },
            }
        return {}

    def _validate_sam_settings(self):
        """Validate SAM settings and show configuration info before enabling Isolate Sound."""
        if not self._input_file:
            QMessageBox.warning(self, "No Input", "Please open an audio file first.")
            return

        desc = self._sam_prompt.text().strip()
        if not desc and not self._masked_video_path:
            QMessageBox.warning(self, "No Prompt",
                                "Please enter a text prompt or generate a visual mask.")
            return

        model_text = self._sam_model_combo.currentText()
        rerank = self._sam_rerank_slider.value()
        is_tv = "TV" in model_text

        # Validate visual prompt constraints
        if self._masked_video_path:
            warnings = []
            if not is_tv:
                warnings.append(
                    "Visual prompting requires a TV model variant "
                    "(e.g. SAM-Audio Large TV).\n"
                    "Non-TV models will ignore the visual mask.")
            if rerank < 2:
                warnings.append(
                    "Visual prompting requires Re-ranking candidates \u2265 2.\n"
                    "The visual ranker picks the best among multiple candidates.")
            if warnings:
                QMessageBox.warning(self, "Visual Prompt Configuration",
                                    "\n\n".join(warnings))
                return

        # Get audio duration
        try:
            import soundfile as sf
            info = sf.info(self._input_file)
            duration_s = info.duration
        except Exception:
            duration_s = 0

        rerank = self._sam_rerank_slider.value()
        chunk_dur = self._chunk_duration_slider.value()
        needs_chunking = duration_s > (chunk_dur + 10)

        # Show/hide chunking settings based on audio length
        self._chunk_group.setVisible(needs_chunking)

        # Build validation summary
        from .sam_backend import get_vram_budget
        budget = get_vram_budget()

        lines = []
        lines.append(f"Audio duration: {duration_s:.1f}s")
        lines.append(f"Re-ranking candidates: {rerank}")

        if budget["total_gb"] > 0:
            # Force a fresh VRAM reading by clearing any cached tensors first
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Re-read after cleanup
                    budget = get_vram_budget()
            except Exception:
                pass

            lines.append(f"GPU VRAM: {budget['free_gb']:.1f}GB free / {budget['total_gb']}GB total")
            est_gb = 4.0 + (rerank * 1.5)
            lines.append(f"Estimated VRAM usage: ~{est_gb:.1f}GB")
            if est_gb > budget["free_gb"]:
                lines.append(f"\nWARNING: Estimated usage exceeds available VRAM!")
                lines.append(f"Recommended max candidates: {budget['max_rerank']}")
                if needs_chunking:
                    lines.append(f"Recommended max chunk duration: {budget['max_chunk_s']}s")

        if needs_chunking:
            overlap = self._chunk_overlap_slider.value() if self._chunk_overlap_cb.isChecked() else 0
            step = chunk_dur - overlap if overlap > 0 else chunk_dur
            n_chunks = max(1, int((duration_s - chunk_dur) / step) + 1)
            lines.append(f"\nChunked processing: {n_chunks} chunks "
                         f"({chunk_dur}s each, {overlap}s overlap)")
        else:
            lines.append(f"\nSingle-pass processing (audio is short enough)")

        model_text = self._sam_model_combo.currentText()
        is_tv = "TV" in model_text

        if self._masked_video_path:
            lines.append(f"\nVisual mask: Ready")
            if not is_tv:
                lines.append("WARNING: Non-TV model selected — visual mask will be ignored!")
                lines.append("Switch to a TV variant (e.g. SAM-Audio Large TV).")
            if rerank < 2:
                lines.append("WARNING: Re-ranking candidates must be \u2265 2 for visual prompting!")
                lines.append("The visual ranker needs multiple candidates to compare.")
        if desc:
            lines.append(f"Text prompt: \"{desc}\"")
        elif self._masked_video_path:
            lines.append("Text prompt: (none — visual-only mode)")
        lines.append(f"Model: {model_text}")

        info_text = "\n".join(lines)

        # Show the Isolate Sound button after validation
        self._separate_btn.setText("Isolate Sound")
        self._separate_btn.setVisible(True)
        self._separate_btn.setEnabled(True)

        reply = QMessageBox.information(
            self, "Settings Validated",
            f"{info_text}\n\nClick 'Isolate Sound' to proceed.",
            QMessageBox.StandardButton.Ok,
        )

    def _on_model_selected(self, current, previous):
        if current is None:
            self._model_desc.setText("")
            self._model_params_group.setVisible(False)
            return
        name = current.data(0, Qt.ItemDataRole.UserRole)
        preset = get_preset_by_name(name)
        if preset:
            stems_str = ", ".join(preset.stems)
            self._model_desc.setText(
                f"{preset.description}\n\nStems: {stems_str}\n"
                f"Architecture: {preset.architecture}"
            )
            self._update_model_params_visibility(preset.architecture)
            self._on_param_preset_changed("Default")

    def _on_rerank_changed(self, value: int):
        self._sam_rerank_label.setText(str(value))
        if value > 1:
            from .sam_backend import get_vram_budget
            budget = get_vram_budget()
            if budget["total_gb"] > 0 and value > budget["max_rerank"]:
                self._vram_warning.setText(
                    f"Warning: {value} candidates may exceed GPU memory "
                    f"({budget['free_gb']}GB free / {budget['total_gb']}GB total)")
                self._vram_warning.setVisible(True)
            else:
                self._vram_warning.setVisible(False)
        else:
            self._vram_warning.setVisible(False)

    def _open_visual_prompt(self):
        """Open the visual prompt dialog for selecting objects in video."""
        if not self._input_file:
            return
        video_path = getattr(self, '_original_input_path', self._input_file)
        if Path(video_path).suffix.lower() not in self._VIDEO_EXTS:
            QMessageBox.information(self, "Video Required",
                                    "Visual prompting requires a video file input.")
            return
        from .visual_prompter import VisualPromptDialog, check_vision_available
        if check_vision_available() is None:
            QMessageBox.warning(self, "No Vision Model",
                "Install ONE of these for visual prompting:\n\n"
                "SAM3 (recommended, no approval needed):\n"
                "  git clone https://github.com/facebookresearch/sam3\n"
                "  cd sam3 && pip install -e .\n\n"
                "SAM2:\n"
                "  git clone https://github.com/facebookresearch/sam2\n"
                "  cd sam2 && pip install -e .\n\n"
                "Do NOT install both at the same time.")
            return
        output_dir = self._get_default_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        dlg = VisualPromptDialog(video_path, output_dir, parent=self)
        if dlg.exec() and dlg.masked_video_path:
            self._masked_video_path = dlg.masked_video_path
            self._visual_status.setText("Mask ready")
            self._visual_status.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 2px;")
            self._clear_mask_btn.setVisible(True)

        # Always free GPU after dialog closes (SAM3 tracker can use ~6GB)
        del dlg
        try:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Visual prompt dialog closed — GPU memory freed")
        except Exception:
            pass

    def _clear_visual_mask(self):
        """Clear the visual prompt mask."""
        self._masked_video_path = None
        self._visual_status.setText("No mask")
        self._visual_status.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
        self._clear_mask_btn.setVisible(False)

    def _load_existing_mask(self):
        """Load a previously saved mask project folder."""
        from PyQt6.QtWidgets import QFileDialog
        from .visual_prompter import validate_mask_folder, load_mask_project

        folder = QFileDialog.getExistingDirectory(
            self, "Select Mask Project Folder",
            str(Path(self._input_file).parent) if self._input_file else "")
        if not folder:
            return

        error = validate_mask_folder(folder)
        if error:
            QMessageBox.warning(self, "Invalid Mask Folder", error)
            return

        result = load_mask_project(folder)
        if result is None:
            QMessageBox.warning(self, "Load Failed", "Could not read mask project.")
            return

        masked_video_path, config = result
        self._masked_video_path = masked_video_path
        source = Path(config.get("source_video", "")).name
        scale = config.get("video_scale", 1)
        n_obj = config.get("num_objects", 1)
        obj_str = f"{n_obj} object{'s' if n_obj != 1 else ''}"
        status = f"Mask loaded ({obj_str}, source: {source}, scale: 1/{scale})"
        self._visual_status.setText(status)
        self._visual_status.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 2px;")
        self._clear_mask_btn.setVisible(True)
        log.info("Loaded mask project from %s", folder)

    def _get_selected_preset(self) -> ModelPreset | None:
        current = self._model_tree.currentItem()
        if current is None:
            return None
        name = current.data(0, Qt.ItemDataRole.UserRole)
        return get_preset_by_name(name)

    def _get_default_output_dir(self) -> str:
        """Return output directory using the active output policy."""
        return self._resolve_output_dir(self._input_file)

    def _get_output_format(self) -> str:
        checked = self._fmt_group.checkedButton()
        if checked:
            return checked.text().lower()
        return "wav"

    # ── Separation (audio-separator) ──

    def _start_separation(self):
        if not self._input_file:
            QMessageBox.warning(self, "No Input", "Please open an audio file first.")
            return

        # Route based on active tab
        if self._main_tabs.currentIndex() == 1:
            self._start_enhancement()
            return

        # Separation tab: route to SAM or standard model
        if self._sub_cat_combo.currentIndex() == 1:
            self._start_sam_separation()
            return

        preset = self._get_selected_preset()
        if not preset:
            QMessageBox.warning(self, "No Model", "Please select a separation model.")
            return

        output_dir = self._get_default_output_dir()
        self._engine.stop()
        self._show_progress("Starting...")

        self._worker = SeparationWorker(
            self._input_file, output_dir, preset, self._get_output_format(),
            extra_params=self._build_separator_params(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_separation_done)
        self._worker.error.connect(self._on_separation_error)
        self._worker.start()

    def _start_batch_separation(self, files: list[str]):
        preset = self._get_selected_preset()
        if not preset:
            QMessageBox.warning(self, "No Model", "Please select a separation model.")
            return
        output_dir = self._resolve_output_dir(files[0], batch=True)
        os.makedirs(output_dir, exist_ok=True)
        self._show_progress("Starting batch...")

        self._batch_worker = BatchSeparationWorker(
            files, output_dir, preset, self._get_output_format()
        )
        self._batch_worker.progress.connect(self._on_progress)
        self._batch_worker.file_finished.connect(self._on_batch_file_done)
        self._batch_worker.all_finished.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    # ── SAM-Audio Separation ──

    def _start_sam_separation(self):
        if not self._input_file:
            QMessageBox.warning(self, "No Input", "Please open an audio file first.")
            return
        if not check_sam_available():
            from .sam_backend import SAM_IMPORT_ERROR
            QMessageBox.warning(self, "SAM-Audio Not Available",
                f"sam-audio could not be loaded.\n\n{SAM_IMPORT_ERROR or 'Not installed.'}\n\n"
                "Install with:\n  pip install git+https://github.com/facebookresearch/sam-audio.git")
            return
        desc = self._sam_prompt.text().strip()
        if not desc and not self._masked_video_path:
            QMessageBox.warning(self, "No Prompt",
                                "Please enter a text prompt or generate a visual mask.")
            return

        sam_models = {
            "SAM-Audio Large": "mrfakename/sam-audio-large",
            "SAM-Audio Base": "mrfakename/sam-audio-base",
            "SAM-Audio Small": "mrfakename/sam-audio-small",
            "SAM-Audio Large TV": "mrfakename/sam-audio-large-tv",
            "SAM-Audio Small TV": "mrfakename/sam-audio-small-tv",
        }
        model_name = sam_models[self._sam_model_combo.currentText()]
        output_dir = self._get_default_output_dir()

        # Check for resumable checkpoint
        from .sam_backend import SAMSeparationWorker, find_checkpoint
        resume_checkpoint = None
        ckpt_path = find_checkpoint(self._input_file, output_dir, desc, model_name)
        if ckpt_path:
            reply = QMessageBox.question(
                self, "Resume Previous Separation",
                "A previous SAM separation was interrupted.\n"
                "Do you want to resume from where it left off?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                resume_checkpoint = ckpt_path

        self._engine.stop()
        self._show_progress(f"Starting SAM-Audio ({self._sam_model_combo.currentText()})...")

        crossfade_type = "hann" if self._crossfade_combo.currentIndex() == 0 else "linear"
        self._worker = SAMSeparationWorker(
            self._input_file, output_dir, desc,
            predict_spans=self._sam_predict_spans.isChecked(),
            reranking_candidates=self._sam_rerank_slider.value(),
            model_name=model_name,
            chunk_duration=self._chunk_duration_slider.value(),
            overlap_duration=self._chunk_overlap_slider.value(),
            overlap_enabled=self._chunk_overlap_cb.isChecked(),
            crossfade_type=crossfade_type,
            resume_checkpoint=resume_checkpoint,
            masked_video_path=self._masked_video_path,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_separation_done)
        self._worker.error.connect(self._on_separation_error)
        self._worker.start()

    # ── Audio Enhancement ──

    def _start_enhancement(self):
        if not self._input_file:
            QMessageBox.warning(self, "No Input", "Please open an audio file first.")
            return

        engine_idx = self._enhance_engine_combo.currentIndex()
        output_dir = self._get_default_output_dir()
        self._engine.stop()

        if engine_idx == 1:
            # DeepFilterNet engine
            if not check_deepfilter_available():
                from .enhance_backend import DEEPFILTER_IMPORT_ERROR
                QMessageBox.warning(
                    self, "DeepFilterNet Not Available",
                    f"deepfilternet could not be loaded.\n\n"
                    f"{DEEPFILTER_IMPORT_ERROR or 'Not installed.'}\n\n"
                    "Install with:\n  pip install deepfilternet")
                return

            self._show_progress("Starting DeepFilterNet enhancement...")
            from .enhance_backend import DeepFilterWorker
            self._worker = DeepFilterWorker(
                input_path=self._input_file,
                output_dir=output_dir,
                atten_lim_db=float(self._df_atten_slider.value()),
                post_filter=self._df_post_filter_cb.isChecked(),
                auto_level=self._enhance_auto_level_cb.isChecked(),
                match_input_loudness=self._enhance_match_level_cb.isChecked(),
                output_gain_db=float(self._enhance_gain_slider.value()),
            )

        elif engine_idx == 2:
            # Studio Sound engine
            remove_noise = self._ss_noise_cb.isChecked()
            remove_reverb = self._ss_reverb_cb.isChecked()
            if not remove_noise and not remove_reverb:
                QMessageBox.warning(
                    self, "Nothing to Do",
                    "Enable at least one Studio Sound stage\n"
                    "(noise removal or reverb removal).")
                return
            if remove_noise and not check_deepfilter_available():
                from .enhance_backend import DEEPFILTER_IMPORT_ERROR
                reply = QMessageBox.question(
                    self, "DeepFilterNet Not Available",
                    f"deepfilternet is not installed — the noise removal stage will be skipped.\n\n"
                    f"{DEEPFILTER_IMPORT_ERROR or 'Not installed.'}\n\n"
                    "Continue with reverb removal only?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                remove_noise = False

            self._show_progress("Starting Studio Sound processing...")
            from .enhance_backend import StudioSoundWorker
            self._worker = StudioSoundWorker(
                input_path=self._input_file,
                output_dir=output_dir,
                remove_noise=remove_noise,
                atten_lim_db=float(self._ss_atten_slider.value()),
                post_filter=self._ss_post_filter_cb.isChecked(),
                remove_reverb=remove_reverb,
                dereverb_segment_size=self._ss_dereverb_segment_slider.value(),
                dereverb_overlap=self._ss_dereverb_overlap_slider.value(),
                dereverb_batch_size=self._ss_dereverb_batch_slider.value(),
                dereverb_pitch_shift=self._ss_dereverb_pitch_slider.value(),
                auto_level=self._enhance_auto_level_cb.isChecked(),
                match_input_loudness=self._enhance_match_level_cb.isChecked(),
                output_gain_db=float(self._enhance_gain_slider.value()),
            )

        elif engine_idx == 3:
            # ClearVoice engine
            if not check_clearvoice_available():
                from .enhance_backend import CLEARVOICE_IMPORT_ERROR
                QMessageBox.warning(
                    self, "ClearVoice Not Available",
                    f"clearvoice could not be loaded.\n\n"
                    f"{CLEARVOICE_IMPORT_ERROR or 'Not installed.'}\n\n"
                    "Install with:\n  pip install clearvoice --no-deps\n\n"
                    "Note: this keeps the existing app dependencies in place."
                )
                return

            self._show_progress("Starting ClearVoice enhancement...")
            from .enhance_backend import ClearVoiceWorker
            self._worker = ClearVoiceWorker(
                input_path=self._input_file,
                output_dir=output_dir,
                model_name=str(self._cv_model_combo.currentData()),
                remove_reverb=self._cv_reverb_cb.isChecked(),
                dereverb_segment_size=self._cv_dereverb_segment_slider.value(),
                dereverb_overlap=self._cv_dereverb_overlap_slider.value(),
                dereverb_batch_size=self._cv_dereverb_batch_slider.value(),
                dereverb_pitch_shift=self._cv_dereverb_pitch_slider.value(),
                auto_level=self._enhance_auto_level_cb.isChecked(),
                match_input_loudness=self._enhance_match_level_cb.isChecked(),
                output_gain_db=float(self._enhance_gain_slider.value()),
            )

        elif engine_idx == 4:
            # MDX-Net engine
            if not check_audio_separator_available():
                from .enhance_backend import AUDIO_SEPARATOR_IMPORT_ERROR
                QMessageBox.warning(
                    self, "MDX-Net Not Available",
                    f"audio-separator could not be loaded.\n\n"
                    f"{AUDIO_SEPARATOR_IMPORT_ERROR or 'Not installed.'}\n\n"
                    "Install with:\n  pip install \"audio-separator[gpu]>=0.25.0\""
                )
                return

            self._show_progress("Starting MDX-Net vocal isolation...")
            from .enhance_backend import MdxNetWorker
            self._worker = MdxNetWorker(
                input_path=self._input_file,
                output_dir=output_dir,
                model_name=str(self._mdx_model_combo.currentData()),
                enable_denoise=self._mdx_denoise_cb.isChecked(),
                mdx_segment_size=self._mdx_segment_slider.value(),
                mdx_overlap=self._mdx_overlap_slider.value() / 100.0,
                mdx_batch_size=self._mdx_batch_slider.value(),
                remove_reverb=self._mdx_reverb_cb.isChecked(),
                dereverb_segment_size=self._mdx_dereverb_segment_slider.value(),
                dereverb_overlap=self._mdx_dereverb_overlap_slider.value(),
                dereverb_batch_size=self._mdx_dereverb_batch_slider.value(),
                dereverb_pitch_shift=self._mdx_dereverb_pitch_slider.value(),
                auto_level=self._enhance_auto_level_cb.isChecked(),
                match_input_loudness=self._enhance_match_level_cb.isChecked(),
                output_gain_db=float(self._enhance_gain_slider.value()),
            )

        elif engine_idx == 5:
            # VoiceFixer — auto-installs inside the worker if needed
            voicefixer_installed = bool(get_backend_probe_state("voicefixer")["installed"])
            self._show_progress(
                "Starting VoiceFixer restoration"
                + (" (will install voicefixer first)..." if not voicefixer_installed else "...")
            )
            from .enhance_backend import VoiceFixerWorker
            self._worker = VoiceFixerWorker(
                input_path=self._input_file,
                output_dir=output_dir,
                mode=self._vf_mode_combo.currentIndex(),
                auto_level=self._enhance_auto_level_cb.isChecked(),
                match_input_loudness=self._enhance_match_level_cb.isChecked(),
                output_gain_db=float(self._enhance_gain_slider.value()),
            )

        elif engine_idx == 6:
            # MetricGAN+ — auto-installs inside the worker if needed
            metricgan_installed = bool(get_backend_probe_state("metricgan")["installed"])
            self._show_progress(
                "Starting MetricGAN+ noise suppression"
                + (" (will install speechbrain first)..." if not metricgan_installed else "...")
            )
            from .enhance_backend import MetricGANWorker
            self._worker = MetricGANWorker(
                input_path=self._input_file,
                output_dir=output_dir,
                auto_level=self._enhance_auto_level_cb.isChecked(),
                match_input_loudness=self._enhance_match_level_cb.isChecked(),
                output_gain_db=float(self._enhance_gain_slider.value()),
            )

        else:
            # Resemble-Enhance engine
            if not check_enhance_available():
                from .enhance_backend import ENHANCE_IMPORT_ERROR
                QMessageBox.warning(
                    self, "Resemble-Enhance Not Available",
                    f"resemble-enhance could not be loaded.\n\n"
                    f"{ENHANCE_IMPORT_ERROR or 'Not installed.'}\n\n"
                    "Install with:\n  pip install resemble-enhance --no-deps")
                return

            mode_idx = self._enhance_mode_combo.currentIndex()
            mode = "enhance" if mode_idx == 0 else "denoise"
            solver = self._cfm_solver_combo.currentText()
            nfe = self._cfm_nfe_slider.value()
            temperature = self._cfm_temp_slider.value() / 100.0
            cfm_guidance = self._cfm_guidance_slider.value() / 100.0
            denoise_before = self._denoise_before_cb.isChecked()

            self._show_progress("Starting Resemble-Enhance...")
            from .enhance_backend import EnhanceWorker
            self._worker = EnhanceWorker(
                input_path=self._input_file,
                output_dir=output_dir,
                mode=mode,
                nfe=nfe,
                solver=solver,
                prior_temperature=temperature,
                cfm_guidance=cfm_guidance,
                denoise_before=denoise_before,
                auto_level=self._enhance_auto_level_cb.isChecked(),
                match_input_loudness=self._enhance_match_level_cb.isChecked(),
                output_gain_db=float(self._enhance_gain_slider.value()),
            )

        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_separation_done)
        self._worker.error.connect(self._on_separation_error)
        self._worker.start()

    # ── Separation Progress/Results ──

    def _show_progress(self, msg: str):
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._progress_label.setVisible(True)
        self._progress_label.setText(msg)
        self._separate_btn.setEnabled(False)
        self._stop_btn.setVisible(True)
        self._stop_btn.setEnabled(True)
        self._output_files.clear()

    def _stop_separation(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.terminate()
        self._hide_progress()
        self._status_bar.showMessage("Separation cancelled.")

    def _on_progress(self, percent: int, message: str):
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)
        self._status_bar.showMessage(message)

    def _on_separation_done(self, output_files: list[str]):
        self._output_files = output_files
        self._hide_progress()
        self._build_stem_tracks(output_files)
        self._status_bar.showMessage(f"Done! {len(output_files)} stems separated.")

    def _on_separation_error(self, error_msg: str):
        self._hide_progress()
        QMessageBox.critical(self, "Separation Error", f"Error:\n\n{error_msg}")
        self._status_bar.showMessage("Error during separation.")

    def _on_batch_file_done(self, input_path: str, output_files: list[str]):
        self._output_files.extend(output_files)

    def _on_batch_done(self):
        self._hide_progress()
        if self._output_files:
            self._build_stem_tracks(self._output_files)
        self._status_bar.showMessage(
            f"Batch complete! {len(self._output_files)} files created."
        )

    def _on_batch_error(self, input_path: str, error_msg: str):
        name = Path(input_path).name if input_path else "Init"
        self._status_bar.showMessage(f"Error on {name}: {error_msg}")

    def _hide_progress(self):
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._separate_btn.setEnabled(True)
        self._stop_btn.setVisible(False)
        self._stop_btn.setEnabled(False)

    # ── Stem Track Building ──

    def _clear_stem_tracks(self):
        for w in self._stem_widgets:
            self._stems_layout.removeWidget(w)
            w.deleteLater()
        self._stem_widgets.clear()
        self._layer_labels.clear()
        if self._original_track_widget:
            self._stems_layout.removeWidget(self._original_track_widget)
            self._original_track_widget.deleteLater()
            self._original_track_widget = None

    def _build_stem_tracks(self, output_files: list[str]):
        self._engine.stop()
        self._clear_effects_session()
        self._clear_stem_tracks()

        # Hide initial waveform, show stem tracks
        self._waveform.setVisible(False)
        self._stems_scroll.setVisible(True)
        self._fx_group.setVisible(True)

        # Determine if this is a SAM separation (for labeling and muting)
        is_sam = (self._main_tabs.currentIndex() == 0 and self._sub_cat_combo.currentIndex() == 1)

        # Load stems into engine
        self._engine.load_stems(output_files)

        insert_pos = 0
        color_idx = 0

        # Original track (muted by default after separation)
        if self._engine.original:
            orig = self._engine.original
            w = StemTrackWidget(-1, "Original sound", orig.data, orig.sample_rate,
                                color_index=color_idx, initially_muted=True)
            w.mute_changed.connect(lambda _, m: self._engine.set_original_muted(m))
            w.seek_requested.connect(self._on_waveform_seek)
            self._stems_layout.insertWidget(insert_pos, w)
            self._original_track_widget = w
            self._layer_labels["original"] = "Original sound"
            self._engine.set_original_muted(True)
            insert_pos += 1
            color_idx += 1

        # Stem tracks
        for i, track in enumerate(self._engine.tracks):
            # Label SAM outputs based on filename, not index
            if is_sam:
                fname = track.file_path.lower()
                if "_target" in fname:
                    label = "Isolated sound"
                elif "_residual" in fname:
                    label = "Without isolated sound"
                else:
                    label = track.name
            else:
                label = track.name
            w = StemTrackWidget(i, label, track.data, track.sample_rate,
                                color_index=color_idx)
            w.mute_changed.connect(self._engine.set_muted)
            w.seek_requested.connect(self._on_waveform_seek)
            self._stems_layout.insertWidget(insert_pos, w)
            self._stem_widgets.append(w)
            self._layer_labels[f"track:{i}"] = label
            insert_pos += 1
            color_idx += 1

        self._transport.set_duration(self._engine.duration)

    # ── Playback ──

    def _on_waveform_seek(self, seconds: float):
        self._engine.seek(seconds)
        self._on_position_changed(seconds)

    def _play_all(self):
        if self._engine.is_playing:
            return
        if self._engine.tracks or self._engine.original:
            # Clear any internal solo
            for t in self._engine.tracks:
                t.solo = False
            if self._engine.original:
                self._engine.original.solo = False
            self._engine.play()
        elif self._input_file:
            self._engine.play_single(self._input_file)

    def _on_position_changed(self, seconds: float):
        self._transport.set_position(seconds)
        # Update initial waveform playhead (before separation)
        if self._waveform.isVisible():
            self._waveform.update_playhead(seconds)
        # Update stem track playheads (after separation)
        if self._original_track_widget:
            self._original_track_widget.update_playhead(seconds)
        for w in self._stem_widgets:
            w.update_playhead(seconds)

    def _on_engine_started(self):
        self._transport.set_playing(True)
        self._status_bar.showMessage("Playing...")

    def _on_engine_stopped(self):
        self._transport.set_playing(False)
        self._status_bar.showMessage("Ready")

    # ── Effects ──

    def _effect_layers(self) -> list[dict]:
        layers: list[dict] = []
        if self._engine.original and self._original_track_widget:
            layers.append(
                {
                    "key": "original",
                    "label": self._layer_labels.get("original", "Original sound"),
                    "track_index": -1,
                    "sample_rate": self._engine.original.sample_rate,
                    "source_audio": (
                        self._engine.original.source_data
                        if self._engine.original.source_data is not None
                        else self._engine.original.data
                    ),
                    "widget": self._original_track_widget,
                }
            )
        for index, track in enumerate(self._engine.tracks):
            widget = self._stem_widgets[index] if index < len(self._stem_widgets) else None
            layers.append(
                {
                    "key": f"track:{index}",
                    "label": self._layer_labels.get(f"track:{index}", track.name),
                    "track_index": index,
                    "sample_rate": track.sample_rate,
                    "source_audio": track.source_data if track.source_data is not None else track.data,
                    "widget": widget,
                }
            )
        return layers

    def _effect_layer_by_key(self, layer_key: str) -> dict | None:
        for layer in self._effect_layers():
            if layer["key"] == layer_key:
                return layer
        return None

    def _apply_processed_audio_to_layer(self, layer: dict, processed: np.ndarray):
        if layer["track_index"] == -1:
            self._engine.update_original_audio(processed)
        else:
            self._engine.update_track_audio(layer["track_index"], processed)
        widget = layer["widget"]
        if widget is not None:
            widget.set_audio_data(processed)

    def _request_effect_preview(self, layer_key: str, state: dict):
        layer = self._effect_layer_by_key(layer_key)
        if layer is None:
            return
        normalized = normalize_effect_state(state)
        self._preview_layer_effect_states[layer_key] = normalized
        self._effects_preview_request_counter += 1
        request_id = self._effects_preview_request_counter
        self._effects_preview_latest_requests[layer_key] = request_id
        self._effects_preview_processor.submit(
            self._effects_preview_session,
            layer_key,
            request_id,
            layer["source_audio"],
            layer["sample_rate"],
            normalized,
        )

    def _on_layer_effects_changed(self, layer_key: str, state: dict):
        layer = self._effect_layer_by_key(layer_key)
        if layer is None:
            return
        self._request_effect_preview(layer_key, state)
        if self._engine.is_playing:
            self._status_bar.showMessage(f"Updating effects for {layer['label']}...")
        else:
            self._status_bar.showMessage(f"Rendering preview for {layer['label']}...")

    def _on_effect_preview_ready(
        self,
        session_id: int,
        layer_key: str,
        request_id: int,
        processed: np.ndarray,
        _state: dict,
    ):
        if session_id != self._effects_preview_session:
            return
        if self._effects_preview_latest_requests.get(layer_key) != request_id:
            return
        layer = self._effect_layer_by_key(layer_key)
        if layer is None:
            return
        self._apply_processed_audio_to_layer(layer, np.asarray(processed, dtype=np.float32))
        if self._engine.is_playing:
            self._status_bar.showMessage(f"Live effects updated for {layer['label']}.")
        else:
            self._status_bar.showMessage(f"Preview updated for {layer['label']}. Press Play to audition.")

    def _on_effect_preview_error(self, session_id: int, layer_key: str, request_id: int, message: str):
        if session_id != self._effects_preview_session:
            return
        if self._effects_preview_latest_requests.get(layer_key) != request_id:
            return
        layer = self._effect_layer_by_key(layer_key)
        label = layer["label"] if layer is not None else layer_key
        self._status_bar.showMessage(f"Could not update effects for {label}.")
        QMessageBox.warning(self, "Effects Preview Failed", f"Could not update {label}:\n\n{message}")

    def _on_effect_preview_busy_changed(self, active: bool):
        if self._effects_dialog is not None:
            self._effects_dialog.set_processing(active)

    def _save_effect_changes(self, states: dict):
        self._layer_effect_states = {
            layer_key: normalize_effect_state(state)
            for layer_key, state in states.items()
        }
        self._preview_layer_effect_states = {
            layer_key: normalize_effect_state(state)
            for layer_key, state in self._layer_effect_states.items()
        }
        for layer in self._effect_layers():
            state = self._layer_effect_states.get(layer["key"], empty_effect_state())
            self._request_effect_preview(layer["key"], state)
        self._status_bar.showMessage("Effects saved for this session.")

    def _discard_effect_changes(self):
        if self._suppress_effect_dialog_discard:
            return
        for layer in self._effect_layers():
            state = self._layer_effect_states.get(layer["key"], empty_effect_state())
            self._request_effect_preview(layer["key"], state)
        self._preview_layer_effect_states = {
            layer["key"]: normalize_effect_state(self._layer_effect_states.get(layer["key"]))
            for layer in self._effect_layers()
        }
        self._status_bar.showMessage("Unsaved effects changes were discarded.")

    def _on_effects_dialog_closed(self):
        self._effects_dialog = None

    def _open_global_effects(self):
        layers = self._effect_layers()
        if not layers:
            return
        if self._effects_dialog is not None:
            self._effects_dialog.raise_()
            self._effects_dialog.activateWindow()
            return
        for layer in layers:
            self._layer_effect_states.setdefault(layer["key"], empty_effect_state())
        self._preview_layer_effect_states = {
            layer["key"]: normalize_effect_state(self._layer_effect_states.get(layer["key"]))
            for layer in layers
        }
        self._effects_dialog = LayerEffectsDialog(layers, self._preview_layer_effect_states, self)
        self._effects_dialog.layer_effects_changed.connect(self._on_layer_effects_changed)
        self._effects_dialog.save_requested.connect(self._save_effect_changes)
        self._effects_dialog.discard_requested.connect(self._discard_effect_changes)
        self._effects_dialog.finished.connect(self._on_effects_dialog_closed)
        self._effects_dialog.show()
        self._effects_dialog.raise_()
        self._effects_dialog.activateWindow()
        self._status_bar.showMessage("Effects panel opened. Use the normal play controls to audition changes.")

    # ── Drag and Drop ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).suffix.lower() in self._AUDIO_EXTS | self._VIDEO_EXTS:
                self._set_input_file(path)
            else:
                self._status_bar.showMessage("Unsupported file format.")

    # ── Dialogs ──

    def _show_about(self):
        QMessageBox.about(self, f"About {APP_DISPLAY_NAME}",
            f"{APP_DISPLAY_NAME} v2.0\n\n"
            "AI-powered audio stem separation using:\n"
            "- python-audio-separator (UVR models)\n"
            "- SAM-Audio (Facebook, text-prompted)\n"
            "- Roformer, MDX-Net, Demucs architectures\n"
            "- GPU-accelerated via ONNX Runtime + CUDA\n\n"
            "Features:\n"
            "- Multi-stem separation & mixing\n"
            "- Per-stem volume & mute controls\n"
            "- Waveform scrubbing\n"
            "- BPM & key detection\n"
            "- Audio effects (EQ, compressor, presets)\n"
            "- Console log viewer"
        )

    def _show_gpu_info(self):
        report = self._acceleration_report or probe_acceleration(self._settings)
        providers = ", ".join(report.providers) if report.providers else "Unavailable"
        info = (
            f"{report.summary}\n\n"
            f"Mode: {report.mode.upper()}\n"
            f"GPU: {report.gpu_name or 'Not detected'}\n"
            f"PyTorch: {report.torch_version or 'Unavailable'}\n"
            f"CUDA: {report.cuda_version or 'Unavailable'}\n"
            f"ONNX Runtime: {report.onnxruntime_version or 'Unavailable'}\n"
            f"Providers: {providers}\n"
            f"FFmpeg: {report.ffmpeg_path or 'Not detected'}\n"
            f"nvidia-smi: {report.nvidia_smi_path or 'Not detected'}\n\n"
            f"{report.details}"
        )
        QMessageBox.information(self, "GPU Information", info)
