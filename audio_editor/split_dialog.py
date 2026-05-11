"""Clip-splitting dialog for trimming audio or video exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .media_utils import export_media_clip, probe_media_duration


def _format_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    minutes, rem_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{minutes}:{secs:02d}.{millis:03d}"


def _time_token(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}{minutes:02d}{secs:02d}_{millis:03d}"


class ClipSelectionWidget(QWidget):
    """Waveform viewer with draggable trim handles."""

    seek_requested = pyqtSignal(float)
    selection_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMouseTracking(True)

        self._mins = np.array([], dtype=np.float32)
        self._maxs = np.array([], dtype=np.float32)
        self._duration = 0.0
        self._selection_start = 0.0
        self._selection_end = 0.0
        self._playhead = 0.0
        self._drag_mode: str | None = None
        self._drag_offset = 0.0
        self._min_selection = 0.05

    @property
    def duration(self) -> float:
        return self._duration

    def selection(self) -> tuple[float, float]:
        return self._selection_start, self._selection_end

    def set_waveform(self, audio_data: np.ndarray, sample_rate: int):
        audio = np.asarray(audio_data, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sample_rate <= 0 or audio.size == 0:
            self._mins = np.array([], dtype=np.float32)
            self._maxs = np.array([], dtype=np.float32)
            self._duration = 0.0
            self._selection_start = 0.0
            self._selection_end = 0.0
            self._playhead = 0.0
            self.update()
            return

        self._duration = float(audio.size / sample_rate)
        bins = min(2400, max(400, int(self.width() * 1.5) if self.width() > 0 else 1200))
        step = max(1, int(np.ceil(audio.size / bins)))
        padded_len = int(np.ceil(audio.size / step) * step)
        if padded_len != audio.size:
            audio = np.pad(audio, (0, padded_len - audio.size))
        reshaped = audio.reshape(-1, step)
        self._mins = reshaped.min(axis=1).astype(np.float32)
        self._maxs = reshaped.max(axis=1).astype(np.float32)

        default_end = min(self._duration, 30.0 if self._duration > 30.0 else self._duration)
        self.set_selection(0.0, max(default_end, min(self._duration, 1.0)), emit_signal=False)
        self.set_playhead(0.0)

    def set_playhead(self, seconds: float):
        if self._duration <= 0:
            self._playhead = 0.0
        else:
            self._playhead = max(0.0, min(float(seconds), self._duration))
        self.update()

    def set_selection(self, start: float, end: float, *, emit_signal: bool = True):
        if self._duration <= 0:
            self._selection_start = 0.0
            self._selection_end = 0.0
            self.update()
            return

        start = max(0.0, min(float(start), self._duration))
        end = max(start + self._min_selection, min(float(end), self._duration))
        if end > self._duration:
            end = self._duration
            start = max(0.0, end - self._min_selection)
        self._selection_start = start
        self._selection_end = end
        self.update()
        if emit_signal:
            self.selection_changed.emit(self._selection_start, self._selection_end)

    def paintEvent(self, event):  # noqa: D401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        rect = self.rect().adjusted(8, 8, -8, -24)
        painter.fillRect(self.rect(), QColor("#1e1e2e"))
        painter.fillRect(rect, QColor("#181825"))
        painter.setPen(QColor("#333344"))
        painter.drawRect(rect)

        if self._duration > 0:
            start_x = self._time_to_x(self._selection_start, rect)
            end_x = self._time_to_x(self._selection_end, rect)

            if start_x > rect.left():
                painter.fillRect(rect.left(), rect.top(), int(start_x - rect.left()), rect.height(), QColor(0, 0, 0, 110))
            if end_x < rect.right():
                painter.fillRect(int(end_x), rect.top(), int(rect.right() - end_x), rect.height(), QColor(0, 0, 0, 110))

            selection_rect_x = int(start_x)
            selection_rect_w = max(2, int(end_x - start_x))
            painter.fillRect(selection_rect_x, rect.top(), selection_rect_w, rect.height(), QColor(137, 180, 250, 55))

        if self._mins.size:
            pen = QPen(QColor("#89b4fa"))
            pen.setWidth(1)
            painter.setPen(pen)
            center_y = rect.center().y()
            amplitude = max(10.0, rect.height() * 0.44)
            denom = max(1, self._mins.size - 1)
            for idx, (mn, mx) in enumerate(zip(self._mins, self._maxs, strict=False)):
                x = rect.left() + (idx / denom) * rect.width()
                top = center_y - float(mx) * amplitude
                bottom = center_y - float(mn) * amplitude
                painter.drawLine(int(x), int(top), int(x), int(bottom))

        if self._duration > 0:
            painter.fillRect(int(start_x) - 3, rect.top(), 6, rect.height(), QColor("#f9e2af"))
            painter.fillRect(int(end_x) - 3, rect.top(), 6, rect.height(), QColor("#f9e2af"))
            playhead_x = self._time_to_x(self._playhead, rect)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(int(playhead_x), rect.top(), int(playhead_x), rect.bottom())

            painter.setPen(QColor("#a6adc8"))
            painter.drawText(rect.left(), self.height() - 6, _format_time(0.0))
            right_text = _format_time(self._duration)
            painter.drawText(rect.right() - 74, self.height() - 6, right_text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._duration > 0 and self._mins.size:
            start, end = self.selection()
            self.set_selection(start, end, emit_signal=False)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._duration <= 0:
            super().mousePressEvent(event)
            return

        rect = self.rect().adjusted(8, 8, -8, -24)
        x = float(event.position().x())
        time = self._x_to_time(x, rect)
        start_x = self._time_to_x(self._selection_start, rect)
        end_x = self._time_to_x(self._selection_end, rect)

        if abs(x - start_x) <= 10:
            self._drag_mode = "start"
        elif abs(x - end_x) <= 10:
            self._drag_mode = "end"
        elif start_x < x < end_x:
            self._drag_mode = "move"
            self._drag_offset = time - self._selection_start
        else:
            self._drag_mode = "seek"
            self.set_playhead(time)
            self.seek_requested.emit(time)
        self._update_cursor(event.position().x(), rect)

    def mouseMoveEvent(self, event):
        rect = self.rect().adjusted(8, 8, -8, -24)
        x = float(event.position().x())
        time = self._x_to_time(x, rect)

        if self._drag_mode == "start":
            self.set_selection(time, self._selection_end)
        elif self._drag_mode == "end":
            self.set_selection(self._selection_start, time)
        elif self._drag_mode == "move":
            length = max(self._min_selection, self._selection_end - self._selection_start)
            new_start = time - self._drag_offset
            new_start = max(0.0, min(new_start, self._duration - length))
            self.set_selection(new_start, new_start + length)
        elif self._drag_mode == "seek":
            self.set_playhead(time)
            self.seek_requested.emit(time)
        else:
            self._update_cursor(x, rect)

    def mouseReleaseEvent(self, event):
        self._drag_mode = None
        rect = self.rect().adjusted(8, 8, -8, -24)
        self._update_cursor(event.position().x(), rect)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

    def _update_cursor(self, x: float, rect):
        if self._duration <= 0:
            self.unsetCursor()
            return
        start_x = self._time_to_x(self._selection_start, rect)
        end_x = self._time_to_x(self._selection_end, rect)
        if abs(x - start_x) <= 10 or abs(x - end_x) <= 10:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        elif start_x < x < end_x:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def _time_to_x(self, seconds: float, rect) -> float:
        if self._duration <= 0:
            return float(rect.left())
        fraction = max(0.0, min(seconds / self._duration, 1.0))
        return rect.left() + fraction * rect.width()

    def _x_to_time(self, x: float, rect) -> float:
        if self._duration <= 0 or rect.width() <= 0:
            return 0.0
        fraction = (x - rect.left()) / rect.width()
        return max(0.0, min(fraction, 1.0)) * self._duration


class SplitClipDialog(QDialog):
    """Trim the currently selected audio or video file and export a clip."""

    def __init__(
        self,
        source_path: str,
        waveform_source_path: str,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self._source_path = source_path
        self._waveform_source_path = waveform_source_path
        self._output_dir = output_dir
        self._source = Path(source_path)
        self._waveform_source = Path(waveform_source_path)
        self._is_video = self._source.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        self._duration = probe_media_duration(source_path)
        self._auto_output_path = True
        self._updating_controls = False
        self._stop_at_selection_end = False

        self.setWindowTitle("Split Clip")
        self.resize(980, 760 if self._is_video else 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        title = QLabel("Split the loaded clip")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Scrub on the waveform, drag the yellow trim handles to resize, or drag the highlighted window to move it."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #a6adc8;")
        layout.addWidget(subtitle)

        source_label = QLabel(f"Source: {self._source.name}")
        source_label.setStyleSheet("color: #89b4fa;")
        source_label.setWordWrap(True)
        layout.addWidget(source_label)

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(0.7)
        self._player.setAudioOutput(self._audio_output)
        self._player.positionChanged.connect(self._on_player_position)
        self._player.durationChanged.connect(self._on_player_duration)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.setSource(QUrl.fromLocalFile(self._source_path))

        if self._is_video:
            self._video_widget = QVideoWidget()
            self._video_widget.setMinimumHeight(260)
            self._video_widget.setStyleSheet("background-color: #181825; border: 1px solid #333344;")
            self._player.setVideoOutput(self._video_widget)
            layout.addWidget(self._video_widget)
        else:
            self._video_widget = None

        self._waveform = ClipSelectionWidget()
        self._waveform.seek_requested.connect(self._seek_to)
        self._waveform.selection_changed.connect(self._on_selection_changed)
        layout.addWidget(self._waveform, stretch=1)

        transport_row = QHBoxLayout()
        self._play_btn = QPushButton("Play Selection")
        self._play_btn.clicked.connect(self._toggle_playback)
        transport_row.addWidget(self._play_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop_playback)
        transport_row.addWidget(self._stop_btn)

        self._time_label = QLabel("0:00.000 / 0:00.000")
        self._time_label.setStyleSheet("font-family: Consolas, 'Courier New', monospace; color: #cdd6f4;")
        transport_row.addWidget(self._time_label)
        transport_row.addStretch()

        self._set_start_btn = QPushButton("Set Start to Playhead")
        self._set_start_btn.clicked.connect(self._set_start_from_playhead)
        transport_row.addWidget(self._set_start_btn)

        self._set_end_btn = QPushButton("Set End to Playhead")
        self._set_end_btn.clicked.connect(self._set_end_from_playhead)
        transport_row.addWidget(self._set_end_btn)

        self._full_clip_btn = QPushButton("Use Full Clip")
        self._full_clip_btn.clicked.connect(self._use_full_clip)
        transport_row.addWidget(self._full_clip_btn)

        layout.addLayout(transport_row)

        selection_group = QGroupBox("Selection")
        selection_layout = QGridLayout(selection_group)
        selection_layout.setHorizontalSpacing(12)
        selection_layout.setVerticalSpacing(10)

        self._start_spin = QDoubleSpinBox()
        self._start_spin.setDecimals(3)
        self._start_spin.setSingleStep(0.1)
        self._start_spin.valueChanged.connect(self._on_spinbox_changed)
        selection_layout.addWidget(QLabel("Start (s)"), 0, 0)
        selection_layout.addWidget(self._start_spin, 0, 1)

        self._end_spin = QDoubleSpinBox()
        self._end_spin.setDecimals(3)
        self._end_spin.setSingleStep(0.1)
        self._end_spin.valueChanged.connect(self._on_spinbox_changed)
        selection_layout.addWidget(QLabel("End (s)"), 0, 2)
        selection_layout.addWidget(self._end_spin, 0, 3)

        self._length_label = QLabel("Length: 0.000 s")
        self._length_label.setStyleSheet("color: #a6adc8;")
        selection_layout.addWidget(self._length_label, 1, 0, 1, 2)

        self._selection_label = QLabel("")
        self._selection_label.setStyleSheet("color: #89b4fa;")
        selection_layout.addWidget(self._selection_label, 1, 2, 1, 2)

        layout.addWidget(selection_group)

        export_group = QGroupBox("Export")
        export_layout = QGridLayout(export_group)
        export_layout.setHorizontalSpacing(12)
        export_layout.setVerticalSpacing(10)

        self._export_video_radio = QRadioButton("Video + audio clip")
        self._export_audio_radio = QRadioButton("Audio-only clip")
        if self._is_video:
            self._export_video_radio.setChecked(True)
            export_layout.addWidget(self._export_video_radio, 0, 0, 1, 2)
            export_layout.addWidget(self._export_audio_radio, 1, 0, 1, 2)
            self._export_video_radio.toggled.connect(self._on_export_mode_changed)
            self._export_audio_radio.toggled.connect(self._on_export_mode_changed)
        else:
            self._export_audio_radio.setChecked(True)
            self._export_audio_radio.setVisible(False)
            audio_only_label = QLabel("This source is audio-only, so the export will be audio-only.")
            audio_only_label.setStyleSheet("color: #a6adc8;")
            export_layout.addWidget(audio_only_label, 0, 0, 1, 2)

        export_layout.addWidget(QLabel("Output"), 2, 0)
        self._output_edit = QLineEdit()
        self._output_edit.textEdited.connect(self._on_output_path_edited)
        export_layout.addWidget(self._output_edit, 2, 1)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse_output)
        export_layout.addWidget(self._browse_btn, 2, 2)

        layout.addWidget(export_group)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self._status_label)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self._export_btn = QPushButton("Export")
        self._export_btn.setObjectName("primaryBtn")
        self._export_btn.clicked.connect(self._export_clip)
        button_row.addWidget(self._export_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(self._cancel_btn)
        layout.addLayout(button_row)

        self._load_waveform()
        self._refresh_time_label(0.0)
        self._update_output_path()

    def closeEvent(self, event):
        self._player.stop()
        super().closeEvent(event)

    def _load_waveform(self):
        try:
            audio, sr = sf.read(self._waveform_source_path, dtype="float32", always_2d=True)
        except Exception as exc:
            QMessageBox.critical(self, "Waveform Load Failed", f"Could not load waveform data:\n\n{exc}")
            self.reject()
            return

        if self._duration <= 0 and sr > 0:
            self._duration = audio.shape[0] / sr
        self._waveform.set_waveform(audio, sr)
        self._start_spin.setRange(0.0, max(0.0, self._duration))
        self._end_spin.setRange(0.0, max(0.0, self._duration))
        start, end = self._waveform.selection()
        self._set_spin_values(start, end)
        self._update_selection_labels(start, end)

    def _toggle_playback(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self._stop_at_selection_end = False
            return

        start, end = self._waveform.selection()
        current = self._player.position() / 1000.0
        if current < start or current >= end:
            self._player.setPosition(int(start * 1000))
        self._stop_at_selection_end = True
        self._player.play()

    def _stop_playback(self):
        self._stop_at_selection_end = False
        self._player.stop()
        start, _ = self._waveform.selection()
        self._seek_to(start)

    def _seek_to(self, seconds: float):
        ms = int(max(0.0, min(seconds, self._duration)) * 1000)
        self._player.setPosition(ms)
        self._waveform.set_playhead(seconds)
        self._refresh_time_label(seconds)

    def _on_player_position(self, position_ms: int):
        seconds = position_ms / 1000.0
        self._waveform.set_playhead(seconds)
        self._refresh_time_label(seconds)
        _, end = self._waveform.selection()
        if self._stop_at_selection_end and seconds >= end:
            self._player.pause()
            self._player.setPosition(int(end * 1000))
            self._stop_at_selection_end = False

    def _on_player_duration(self, duration_ms: int):
        if duration_ms > 0 and abs((duration_ms / 1000.0) - self._duration) > 0.05:
            self._duration = duration_ms / 1000.0
            self._start_spin.setRange(0.0, self._duration)
            self._end_spin.setRange(0.0, self._duration)
        self._refresh_time_label(self._player.position() / 1000.0)

    def _on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("Pause")
        else:
            self._play_btn.setText("Play Selection")

    def _on_selection_changed(self, start: float, end: float):
        self._set_spin_values(start, end)
        self._update_selection_labels(start, end)
        if self._auto_output_path:
            self._update_output_path()

    def _on_spinbox_changed(self):
        if self._updating_controls:
            return
        start = self._start_spin.value()
        end = self._end_spin.value()
        if end <= start:
            if self.sender() is self._start_spin:
                end = min(self._duration, start + 0.05)
            else:
                start = max(0.0, end - 0.05)
        self._waveform.set_selection(start, end, emit_signal=True)

    def _set_spin_values(self, start: float, end: float):
        self._updating_controls = True
        self._start_spin.setValue(start)
        self._end_spin.setValue(end)
        self._updating_controls = False

    def _update_selection_labels(self, start: float, end: float):
        self._length_label.setText(f"Length: {max(0.0, end - start):.3f} s")
        self._selection_label.setText(f"{_format_time(start)} to {_format_time(end)}")

    def _refresh_time_label(self, seconds: float):
        self._time_label.setText(f"{_format_time(seconds)} / {_format_time(self._duration)}")

    def _set_start_from_playhead(self):
        current = self._player.position() / 1000.0
        _, end = self._waveform.selection()
        self._waveform.set_selection(current, end)

    def _set_end_from_playhead(self):
        current = self._player.position() / 1000.0
        start, _ = self._waveform.selection()
        self._waveform.set_selection(start, current)

    def _use_full_clip(self):
        self._waveform.set_selection(0.0, self._duration)
        self._seek_to(0.0)

    def _is_audio_only_export(self) -> bool:
        return (not self._is_video) or self._export_audio_radio.isChecked()

    def _on_export_mode_changed(self):
        if self._auto_output_path:
            self._update_output_path()

    def _on_output_path_edited(self, _text: str):
        self._auto_output_path = False

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Clip",
            self._output_edit.text().strip() or self._build_default_output_path(),
            self._save_file_filter(),
        )
        if path:
            self._auto_output_path = False
            self._output_edit.setText(path)

    def _save_file_filter(self) -> str:
        if not self._is_audio_only_export():
            return "MP4 Video (*.mp4);;All Files (*)"
        return "WAV Audio (*.wav);;MP3 Audio (*.mp3);;M4A Audio (*.m4a);;FLAC Audio (*.flac);;All Files (*)"

    def _build_default_output_path(self) -> str:
        start, end = self._waveform.selection()
        suffix = ".wav" if self._is_audio_only_export() else ".mp4"
        output_dir = Path(self._output_dir) if self._output_dir else self._source.parent
        name = f"{self._source.stem}_clip_{_time_token(start)}_{_time_token(end)}{suffix}"
        return str(output_dir / name)

    def _update_output_path(self):
        self._output_edit.setText(self._build_default_output_path())
        self._auto_output_path = True

    def _normalize_output_path(self) -> Path:
        path_text = self._output_edit.text().strip() or self._build_default_output_path()
        output = Path(path_text)
        if self._is_audio_only_export():
            if output.suffix.lower() not in {".wav", ".mp3", ".m4a", ".aac", ".flac"}:
                output = output.with_suffix(".wav")
        elif output.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
            output = output.with_suffix(".mp4")
        self._output_edit.setText(str(output))
        return output

    def _export_clip(self):
        start, end = self._waveform.selection()
        if end - start < 0.05:
            QMessageBox.warning(self, "Selection Too Short", "Choose a longer range before exporting.")
            return

        output = self._normalize_output_path()
        if output.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite Existing File?",
                f"{output.name} already exists.\n\nDo you want to overwrite it?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._status_label.setText("Exporting clip...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export_media_clip(
                self._source_path,
                str(output),
                start,
                end,
                audio_only=self._is_audio_only_export(),
            )
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "FFmpeg Not Found", str(exc))
            self._status_label.setText("Export failed: FFmpeg not found.")
            return
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            self._status_label.setText("Export failed.")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._status_label.setText(f"Exported: {output}")
        QMessageBox.information(self, "Clip Exported", f"Saved clip to:\n\n{output}")
        self.accept()
