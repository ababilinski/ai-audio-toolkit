"""Modeless layered audio effects panel."""

from __future__ import annotations

from copy import deepcopy

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .effects import (
    EFFECT_KEYS_BY_SECTION,
    EFFECT_SECTION_ORDER,
    EFFECT_SPEC_MAP,
    empty_effect_state,
    normalize_effect_state,
)


class _ScrollFriendlySlider(QSlider):
    """Allow mouse-wheel scrolling to move the panel instead of hijacking focus."""

    def wheelEvent(self, event):
        event.ignore()


class LayerEffectsDialog(QDialog):
    """Live effects panel with per-layer tabs and explicit save/discard semantics."""

    layer_effects_changed = pyqtSignal(str, dict)
    save_requested = pyqtSignal(dict)
    discard_requested = pyqtSignal()

    def __init__(self, layers: list[dict], effect_states: dict[str, dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add sound effects")
        self.setModal(False)
        self.resize(540, 760)
        self.setMinimumSize(480, 620)
        self._tab_keys: list[str] = []
        self._controls: dict[str, dict[str, dict[str, object]]] = {}
        self._timers: dict[str, QTimer] = {}
        self._states = {
            layer["key"]: normalize_effect_state(effect_states.get(layer["key"]))
            for layer in layers
        }
        self._committing = False

        layout = QVBoxLayout(self)
        header = QLabel(
            "Use the main transport controls to audition changes while this panel is open. "
            "Edits update the visible waveform and playback live. Click Save Changes to keep them, "
            "or Close to discard them. Vocal enhancer sliders support -1.00 to +1.00."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #a6adc8; padding: 2px 4px 8px 4px;")
        layout.addWidget(header)

        self._busy_row = QWidget()
        busy_layout = QHBoxLayout(self._busy_row)
        busy_layout.setContentsMargins(4, 0, 4, 8)
        busy_layout.setSpacing(8)
        self._busy_label = QLabel("Applying effect preview...")
        self._busy_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
        busy_layout.addWidget(self._busy_label)
        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedHeight(10)
        busy_layout.addWidget(self._busy_bar, stretch=1)
        self._busy_row.setVisible(False)
        layout.addWidget(self._busy_row)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        for layer in layers:
            self._tab_keys.append(layer["key"])
            self._controls[layer["key"]] = {}
            self._tabs.addTab(self._build_layer_tab(layer["key"], layer["label"]), layer["label"])

        btn_row = QHBoxLayout()
        self._reset_layer_btn = QPushButton("Reset Selected Layer")
        self._reset_layer_btn.clicked.connect(self._reset_current_layer)
        btn_row.addWidget(self._reset_layer_btn)

        self._reset_all_btn = QPushButton("Reset All")
        self._reset_all_btn.clicked.connect(self._reset_all_layers)
        btn_row.addWidget(self._reset_all_btn)

        btn_row.addStretch()

        self._save_btn = QPushButton("Save Changes")
        self._save_btn.setObjectName("primaryBtn")
        self._save_btn.clicked.connect(self._save_and_close)
        btn_row.addWidget(self._save_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

    def _build_layer_tab(self, layer_key: str, layer_label: str) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        info = QLabel(f"Layer: {layer_label}")
        info.setStyleSheet("font-weight: bold; color: #cdd6f4;")
        root.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_body = QWidget()
        scroll_layout = QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        for section in EFFECT_SECTION_ORDER:
            group = QGroupBox(section)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(8)
            for effect_key in EFFECT_KEYS_BY_SECTION[section]:
                group_layout.addWidget(self._build_effect_row(layer_key, effect_key))
            scroll_layout.addWidget(group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_body)
        root.addWidget(scroll, stretch=1)
        return page

    def _build_effect_row(self, layer_key: str, effect_key: str) -> QWidget:
        spec = EFFECT_SPEC_MAP[effect_key]
        state = self._states.get(layer_key, empty_effect_state())[effect_key]

        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        top = QHBoxLayout()
        check = QCheckBox(spec.label)
        check.setChecked(bool(state["enabled"]))
        check.toggled.connect(
            lambda checked, lk=layer_key, ek=effect_key: self._on_enabled_changed(lk, ek, checked)
        )
        top.addWidget(check)
        top.addStretch()
        value = QLabel(f"{float(state['amount']):+.2f}")
        value.setFixedWidth(48)
        top.addWidget(value)
        layout.addLayout(top)

        desc = QLabel(spec.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a6adc8; font-size: 10px;")
        layout.addWidget(desc)

        slider = _ScrollFriendlySlider(Qt.Orientation.Horizontal)
        slider.setRange(self._slider_min(spec), self._slider_max(spec))
        slider.setValue(self._amount_to_slider_value(spec, float(state["amount"])))
        slider.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        slider.valueChanged.connect(
            lambda raw, lk=layer_key, ek=effect_key, lbl=value: self._on_amount_changed(lk, ek, raw, lbl)
        )
        layout.addWidget(slider)

        self._controls[layer_key][effect_key] = {
            "check": check,
            "slider": slider,
            "label": value,
        }
        return row

    def _current_layer_key(self) -> str | None:
        index = self._tabs.currentIndex()
        if index < 0 or index >= len(self._tab_keys):
            return None
        return self._tab_keys[index]

    def _slider_min(self, spec) -> int:
        return int(round(spec.min_amount * 100))

    def _slider_max(self, spec) -> int:
        return int(round(spec.max_amount * 100))

    def _slider_value_to_amount(self, spec, raw_value: int) -> float:
        return max(spec.min_amount, min(spec.max_amount, raw_value / 100.0))

    def _amount_to_slider_value(self, spec, amount: float) -> int:
        return int(round(max(spec.min_amount, min(spec.max_amount, amount)) * 100.0))

    def _collect_layer_state(self, layer_key: str) -> dict[str, dict[str, float | bool]]:
        state = empty_effect_state()
        for effect_key, controls in self._controls[layer_key].items():
            check = controls["check"]
            slider = controls["slider"]
            spec = EFFECT_SPEC_MAP[effect_key]
            state[effect_key]["enabled"] = bool(check.isChecked())
            state[effect_key]["amount"] = self._slider_value_to_amount(spec, slider.value())
        return state

    def states_snapshot(self) -> dict[str, dict[str, dict[str, float | bool]]]:
        return {
            layer_key: deepcopy(self._collect_layer_state(layer_key))
            for layer_key in self._tab_keys
        }

    def _schedule_emit(self, layer_key: str) -> None:
        self.set_processing(True)
        timer = self._timers.get(layer_key)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda lk=layer_key: self._emit_layer_state(lk))
            self._timers[layer_key] = timer
        timer.start(60)

    def _emit_layer_state(self, layer_key: str) -> None:
        state = self._collect_layer_state(layer_key)
        self._states[layer_key] = deepcopy(state)
        self.layer_effects_changed.emit(layer_key, deepcopy(state))

    def _on_enabled_changed(self, layer_key: str, effect_key: str, checked: bool) -> None:
        controls = self._controls[layer_key][effect_key]
        slider = controls["slider"]
        spec = EFFECT_SPEC_MAP[effect_key]
        if checked and slider.value() == self._amount_to_slider_value(spec, 0.0):
            slider.setValue(self._amount_to_slider_value(spec, spec.default_amount))
        self._schedule_emit(layer_key)

    def _on_amount_changed(self, layer_key: str, effect_key: str, raw_value: int, value_label: QLabel) -> None:
        spec = EFFECT_SPEC_MAP[effect_key]
        value_label.setText(f"{self._slider_value_to_amount(spec, raw_value):+.2f}")
        if raw_value != 0 and not self._controls[layer_key][effect_key]["check"].isChecked():
            self._controls[layer_key][effect_key]["check"].setChecked(True)
            return
        self._schedule_emit(layer_key)

    def _set_layer_state(
        self,
        layer_key: str,
        state: dict[str, dict[str, float | bool]],
        *,
        emit: bool,
    ) -> None:
        normalized = normalize_effect_state(state)
        for effect_key, controls in self._controls[layer_key].items():
            check = controls["check"]
            slider = controls["slider"]
            label = controls["label"]
            spec = EFFECT_SPEC_MAP[effect_key]
            check.blockSignals(True)
            slider.blockSignals(True)
            check.setChecked(bool(normalized[effect_key]["enabled"]))
            slider.setValue(self._amount_to_slider_value(spec, float(normalized[effect_key]["amount"])))
            label.setText(f"{float(normalized[effect_key]['amount']):+.2f}")
            slider.blockSignals(False)
            check.blockSignals(False)
        self._states[layer_key] = deepcopy(normalized)
        if emit:
            self.layer_effects_changed.emit(layer_key, deepcopy(normalized))

    def _reset_current_layer(self) -> None:
        layer_key = self._current_layer_key()
        if layer_key:
            self._set_layer_state(layer_key, empty_effect_state(), emit=True)

    def _reset_all_layers(self) -> None:
        for layer_key in self._tab_keys:
            self._set_layer_state(layer_key, empty_effect_state(), emit=True)

    def _save_and_close(self) -> None:
        self._committing = True
        self.save_requested.emit(self.states_snapshot())
        self.close()

    def set_processing(self, active: bool) -> None:
        self._busy_row.setVisible(active)
        if active:
            self._busy_label.setText("Applying effect preview...")
        else:
            self._busy_label.setText("")

    def is_processing(self) -> bool:
        return self._busy_row.isVisible()

    def closeEvent(self, event):
        if not self._committing:
            self.discard_requested.emit()
        super().closeEvent(event)
