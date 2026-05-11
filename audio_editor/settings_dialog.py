"""Application settings dialog."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .diagnostics import guidance_for_report, probe_acceleration
from .runtime import ensure_ffmpeg_environment, ensure_runtime_dll_directories
from .settings import (
    AppSettings,
    ModelEntry,
    _discover_deepfilter_models,
    _discover_hf_models,
    _discover_separator_models,
    delete_model_entry,
    get_app_settings,
    legacy_data_root,
)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{num_bytes} B"


class SettingsDialog(QDialog):
    """Dialog for managing persistent application settings."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        initial_tab: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings or get_app_settings()
        self._current_entries: dict[str, ModelEntry] = {}

        self.setWindowTitle("Settings")
        self.resize(860, 700)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())

        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._tabs.addTab(self._build_general_tab(), "General")
        self._tabs.addTab(self._build_models_tab(), "Models & Storage")
        self._tabs.addTab(self._build_acceleration_tab(), "Acceleration")
        self._tabs.addTab(self._build_paths_tab(), "Paths & Tools")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_from_settings()
        self._refresh_model_entries()
        self._refresh_acceleration()

        tab_index = {
            "general": 0,
            "models": 1,
            "acceleration": 2,
            "paths": 3,
        }.get((initial_tab or "").lower())
        if tab_index is not None:
            self._tabs.setCurrentIndex(tab_index)

    def _build_general_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        output_group = QGroupBox("Output Defaults")
        output_layout = QFormLayout(output_group)
        self._output_mode_combo = QComboBox()
        self._output_mode_combo.addItem("Per input folder", "per_input")
        self._output_mode_combo.addItem("Fixed default folder", "fixed")
        output_layout.addRow("Output mode:", self._output_mode_combo)

        output_row = QHBoxLayout()
        self._default_output_edit = QLineEdit()
        browse_output = QPushButton("Browse...")
        browse_output.clicked.connect(lambda: self._browse_directory(self._default_output_edit))
        output_row.addWidget(self._default_output_edit, stretch=1)
        output_row.addWidget(browse_output)
        output_layout.addRow("Default output folder:", output_row)

        self._reuse_last_folder_cb = QCheckBox("Reuse the last folder used in file pickers")
        output_layout.addRow("", self._reuse_last_folder_cb)
        layout.addWidget(output_group)

        runtime_group = QGroupBox("Application Data")
        runtime_layout = QVBoxLayout(runtime_group)
        self._settings_path_label = QLabel("")
        self._settings_path_label.setWordWrap(True)
        runtime_layout.addWidget(self._settings_path_label)

        legacy = legacy_data_root()
        if legacy.exists():
            legacy_label = QLabel(
                "Legacy data was detected and adopted where possible:\n"
                f"{legacy}"
            )
            legacy_label.setWordWrap(True)
            runtime_layout.addWidget(legacy_label)
        layout.addWidget(runtime_group)
        layout.addStretch()
        return page

    def _build_models_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        paths_group = QGroupBox("Managed Model Locations")
        paths_layout = QFormLayout(paths_group)
        self._separator_models_edit = QLineEdit()
        self._hf_cache_edit = QLineEdit()
        self._deepfilter_cache_edit = QLineEdit()
        paths_layout.addRow("Separator models:", self._path_row(self._separator_models_edit))
        paths_layout.addRow("Hugging Face cache:", self._path_row(self._hf_cache_edit))
        paths_layout.addRow("DeepFilterNet cache:", self._path_row(self._deepfilter_cache_edit))
        layout.addWidget(paths_group)

        inventory_group = QGroupBox("Downloaded Models")
        inventory_layout = QVBoxLayout(inventory_group)
        self._model_summary_label = QLabel("")
        inventory_layout.addWidget(self._model_summary_label)

        self._model_tree = QTreeWidget()
        self._model_tree.setColumnCount(4)
        self._model_tree.setHeaderLabels(["Backend", "Name", "Size", "Path"])
        self._model_tree.setRootIsDecorated(False)
        self._model_tree.setAlternatingRowColors(True)
        self._model_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        inventory_layout.addWidget(self._model_tree, stretch=1)

        action_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_model_entries)
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self._open_selected_model)
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._delete_selected_models)
        clear_btn = QPushButton("Delete All Downloaded Models")
        clear_btn.clicked.connect(self._delete_all_models)
        action_row.addWidget(refresh_btn)
        action_row.addWidget(open_btn)
        action_row.addWidget(delete_btn)
        action_row.addWidget(clear_btn)
        action_row.addStretch()
        inventory_layout.addLayout(action_row)
        layout.addWidget(inventory_group, stretch=1)
        return page

    def _build_acceleration_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        summary_group = QGroupBox("Acceleration Status")
        summary_layout = QVBoxLayout(summary_group)
        self._acc_summary_label = QLabel("")
        self._acc_summary_label.setWordWrap(True)
        summary_layout.addWidget(self._acc_summary_label)

        details_grid = QGridLayout()
        self._acc_value_labels: dict[str, QLabel] = {}
        rows = [
            ("Mode", "mode"),
            ("GPU", "gpu"),
            ("PyTorch", "torch"),
            ("CUDA", "cuda"),
            ("ONNX Runtime", "ort"),
            ("Providers", "providers"),
            ("FFmpeg", "ffmpeg"),
            ("nvidia-smi", "nvidia_smi"),
        ]
        for row, (title, key) in enumerate(rows):
            details_grid.addWidget(QLabel(f"{title}:"), row, 0)
            label = QLabel("")
            label.setWordWrap(True)
            details_grid.addWidget(label, row, 1)
            self._acc_value_labels[key] = label
        summary_layout.addLayout(details_grid)
        layout.addWidget(summary_group)

        guidance_group = QGroupBox("Guidance")
        guidance_layout = QVBoxLayout(guidance_group)
        self._acc_guidance = QPlainTextEdit()
        self._acc_guidance.setReadOnly(True)
        self._acc_guidance.setMinimumHeight(220)
        guidance_layout.addWidget(self._acc_guidance)

        action_row = QHBoxLayout()
        recheck_btn = QPushButton("Re-check GPU Setup")
        recheck_btn.clicked.connect(self._refresh_acceleration)
        driver_btn = QPushButton("NVIDIA Drivers")
        driver_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://www.nvidia.com/Download/index.aspx"))
        )
        cuda_btn = QPushButton("CUDA Toolkit")
        cuda_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://developer.nvidia.com/cuda-downloads"))
        )
        torch_btn = QPushButton("PyTorch Install Guide")
        torch_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://pytorch.org/get-started/locally/"))
        )
        action_row.addWidget(recheck_btn)
        action_row.addWidget(driver_btn)
        action_row.addWidget(cuda_btn)
        action_row.addWidget(torch_btn)
        action_row.addStretch()
        guidance_layout.addLayout(action_row)

        layout.addWidget(guidance_group, stretch=1)
        return page

    def _build_paths_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        tools_group = QGroupBox("External Tools")
        tools_layout = QFormLayout(tools_group)
        self._ffmpeg_override_edit = QLineEdit()
        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self._ffmpeg_override_edit, stretch=1)
        ffmpeg_browse = QPushButton("Browse...")
        ffmpeg_browse.clicked.connect(self._browse_ffmpeg)
        ffmpeg_row.addWidget(ffmpeg_browse)
        tools_layout.addRow("FFmpeg override:", ffmpeg_row)
        layout.addWidget(tools_group)

        runtime_group = QGroupBox("Additional Runtime Search Paths")
        runtime_layout = QVBoxLayout(runtime_group)
        runtime_layout.addWidget(
            QLabel(
                "One path per line. These paths are prepended to PATH and the Windows DLL search order.\n"
                "Use this only when the app cannot discover a required runtime path automatically."
            )
        )
        self._extra_runtime_edit = QPlainTextEdit()
        self._extra_runtime_edit.setPlaceholderText(
            "C:\\path\\to\\custom\\dlls\nC:\\path\\to\\other\\runtime"
        )
        runtime_layout.addWidget(self._extra_runtime_edit)
        layout.addWidget(runtime_group, stretch=1)
        return page

    def _path_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, stretch=1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse_directory(line_edit))
        row.addWidget(browse)
        return widget

    def _load_from_settings(self) -> None:
        self._output_mode_combo.setCurrentIndex(
            1 if self._settings.output_mode() == "fixed" else 0
        )
        self._default_output_edit.setText(self._settings.default_output_dir())
        self._reuse_last_folder_cb.setChecked(self._settings.reuse_last_used_folder())
        self._separator_models_edit.setText(self._settings.separator_model_dir())
        self._hf_cache_edit.setText(self._settings.hf_cache_dir())
        self._deepfilter_cache_edit.setText(self._settings.deepfilter_model_dir())
        self._ffmpeg_override_edit.setText(self._settings.ffmpeg_override())
        self._extra_runtime_edit.setPlainText("\n".join(self._settings.extra_runtime_paths()))
        self._settings_path_label.setText(f"Settings file:\n{self._settings.path}")

    def _browse_directory(self, line_edit: QLineEdit) -> None:
        start = line_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if path:
            line_edit.setText(path)

    def _browse_ffmpeg(self) -> None:
        start = self._ffmpeg_override_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FFmpeg Executable",
            start,
            "FFmpeg (ffmpeg.exe);;Executables (*.exe);;All Files (*)",
        )
        if path:
            self._ffmpeg_override_edit.setText(path)

    def _selected_entries(self) -> list[ModelEntry]:
        entries: list[ModelEntry] = []
        for item in self._model_tree.selectedItems():
            path = item.data(0, Qt.ItemDataRole.UserRole)
            if path and path in self._current_entries:
                entries.append(self._current_entries[path])
        return entries

    def _model_entries_from_controls(self) -> list[ModelEntry]:
        entries: list[ModelEntry] = []
        separator_root = self._separator_models_edit.text().strip()
        hf_root = self._hf_cache_edit.text().strip()
        deepfilter_root = self._deepfilter_cache_edit.text().strip()
        if separator_root:
            entries.extend(_discover_separator_models(Path(separator_root)))
        if hf_root:
            entries.extend(_discover_hf_models(Path(hf_root)))
        if deepfilter_root:
            entries.extend(_discover_deepfilter_models(Path(deepfilter_root)))
        return entries

    def _refresh_model_entries(self) -> None:
        self._model_tree.clear()
        self._current_entries.clear()
        entries = self._model_entries_from_controls()
        total_size = 0
        for entry in entries:
            total_size += entry.size_bytes
            item = QTreeWidgetItem(
                [
                    entry.backend,
                    entry.name,
                    _human_size(entry.size_bytes),
                    str(entry.path),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(entry.path))
            self._model_tree.addTopLevelItem(item)
            self._current_entries[str(entry.path)] = entry
        self._model_tree.resizeColumnToContents(0)
        self._model_tree.resizeColumnToContents(1)
        self._model_tree.resizeColumnToContents(2)
        self._model_summary_label.setText(
            f"Found {len(entries)} downloaded model entries using {_human_size(total_size)}."
        )

    def _open_selected_model(self) -> None:
        entries = self._selected_entries()
        if not entries:
            QMessageBox.information(self, "Open Model", "Select a model entry first.")
            return
        target = entries[0].path if entries[0].path.is_dir() else entries[0].path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _delete_selected_models(self) -> None:
        entries = self._selected_entries()
        if not entries:
            QMessageBox.information(self, "Delete Models", "Select one or more model entries first.")
            return

        paths = "\n".join(str(entry.path) for entry in entries)
        reply = QMessageBox.question(
            self,
            "Delete Selected Models",
            "Delete the selected downloaded model files/folders?\n\n"
            f"{paths}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []
        for entry in entries:
            try:
                delete_model_entry(entry)
            except OSError as exc:
                errors.append(f"{entry.path}: {exc}")
        self._refresh_model_entries()
        if errors:
            QMessageBox.warning(self, "Delete Models", "\n".join(errors))

    def _delete_all_models(self) -> None:
        entries = self._model_entries_from_controls()
        if not entries:
            QMessageBox.information(self, "Delete Models", "No downloaded models were found.")
            return

        roots = [
            self._separator_models_edit.text().strip(),
            self._hf_cache_edit.text().strip(),
            self._deepfilter_cache_edit.text().strip(),
        ]
        reply = QMessageBox.question(
            self,
            "Delete All Downloaded Models",
            "Delete every discovered downloaded model entry under these roots?\n\n"
            + "\n".join(root for root in roots if root),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        errors: list[str] = []
        for entry in entries:
            try:
                delete_model_entry(entry)
            except OSError as exc:
                errors.append(f"{entry.path}: {exc}")
        self._refresh_model_entries()
        if errors:
            QMessageBox.warning(self, "Delete Models", "\n".join(errors))

    def _refresh_acceleration(self) -> None:
        report = probe_acceleration(self._settings)
        self._acc_summary_label.setText(report.summary)
        color = "#a6e3a1" if report.mode == "gpu" else "#f9e2af"
        if report.reason_code not in {"ok", "no_nvidia_gpu", "ffmpeg_missing"}:
            color = "#f38ba8"
        self._acc_summary_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        self._acc_value_labels["mode"].setText(report.mode.upper())
        self._acc_value_labels["gpu"].setText(report.gpu_name or "Not detected")
        self._acc_value_labels["torch"].setText(report.torch_version or "Unavailable")
        self._acc_value_labels["cuda"].setText(report.cuda_version or "Unavailable")
        self._acc_value_labels["ort"].setText(report.onnxruntime_version or "Unavailable")
        self._acc_value_labels["providers"].setText(", ".join(report.providers) if report.providers else "Unavailable")
        self._acc_value_labels["ffmpeg"].setText(report.ffmpeg_path or "Not detected")
        self._acc_value_labels["nvidia_smi"].setText(report.nvidia_smi_path or "Not detected")
        self._acc_guidance.setPlainText(
            guidance_for_report(report)
            + "\n\nDiagnostics:\n"
            + report.details
        )

    def _apply_values_to_settings(self, *, sync: bool) -> None:
        mode = self._output_mode_combo.currentData()
        self._settings.set_output_mode(mode)
        self._settings.set_default_output_dir(self._default_output_edit.text())
        self._settings.set_reuse_last_used_folder(self._reuse_last_folder_cb.isChecked())
        self._settings.set_separator_model_dir(self._separator_models_edit.text())
        self._settings.set_hf_cache_dir(self._hf_cache_edit.text())
        self._settings.set_deepfilter_model_dir(self._deepfilter_cache_edit.text())
        self._settings.set_ffmpeg_override(self._ffmpeg_override_edit.text())
        extra_paths = [
            line.strip()
            for line in self._extra_runtime_edit.toPlainText().splitlines()
            if line.strip()
        ]
        self._settings.set_extra_runtime_paths(extra_paths)
        if sync:
            self._settings.sync()

    def _save_and_accept(self) -> None:
        if self._output_mode_combo.currentData() == "fixed" and not self._default_output_edit.text().strip():
            QMessageBox.warning(
                self,
                "Missing Output Folder",
                "Choose a default output folder or switch the output mode back to 'Per input folder'.",
            )
            self._tabs.setCurrentIndex(0)
            return

        self._apply_values_to_settings(sync=True)
        self._settings.apply_runtime_environment()
        ensure_runtime_dll_directories()
        ensure_ffmpeg_environment()
        self.accept()
