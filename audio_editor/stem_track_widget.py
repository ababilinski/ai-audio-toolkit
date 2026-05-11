"""Individual stem track widget matching SAM-Audio reference design:
[Speaker/Mute icon] [Colored container with label overlay + waveform]
"""
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSizePolicy,
)


# Track color palette (hue, border color, fill color, muted fill)
TRACK_COLORS = [
    {"border": "#5b8a72", "fill": "#2d4a3e", "wave": "#7ec9a3", "name": "teal"},
    {"border": "#c76d8f", "fill": "#4a2d3e", "wave": "#e88aaf", "name": "pink"},
    {"border": "#6b7fc7", "fill": "#2d3250", "wave": "#8a9ee8", "name": "blue"},
    {"border": "#c7a86b", "fill": "#4a3d2d", "wave": "#e8c88a", "name": "gold"},
    {"border": "#8b6bc7", "fill": "#3a2d50", "wave": "#aa8ae8", "name": "purple"},
    {"border": "#6bc7b8", "fill": "#2d4a45", "wave": "#8ae8d8", "name": "cyan"},
]


def get_track_color(index: int) -> dict:
    return TRACK_COLORS[index % len(TRACK_COLORS)]


class _SpeakerButton(QPushButton):
    """Custom-painted speaker/mute button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._muted = False
        self.setFixedSize(40, 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)

    @property
    def muted(self):
        return self._muted

    def set_muted(self, muted: bool):
        self._muted = muted
        self.setChecked(muted)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() / 2, self.height() / 2
        color = QColor("#cdd6f4")
        p.setPen(QPen(color, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Offset left so icon + waves fit within button
        ox = cx - 4

        # Speaker body
        p.drawRect(QRectF(ox - 5, cy - 3, 4, 6))
        # Speaker cone
        path = QPainterPath()
        path.moveTo(ox - 1, cy - 3)
        path.lineTo(ox + 3, cy - 6)
        path.lineTo(ox + 3, cy + 6)
        path.lineTo(ox - 1, cy + 3)
        path.closeSubpath()
        p.drawPath(path)

        if self._muted:
            # Draw X
            p.setPen(QPen(QColor("#f38ba8"), 2.0))
            p.drawLine(QPointF(ox + 5, cy - 4), QPointF(ox + 11, cy + 4))
            p.drawLine(QPointF(ox + 5, cy + 4), QPointF(ox + 11, cy - 4))
        else:
            # Draw sound waves
            p.setPen(QPen(color, 1.2))
            for r in [4, 7]:
                p.drawArc(QRectF(ox + 3, cy - r, r * 2, r * 2), -45 * 16, 90 * 16)

        p.end()


class WaveformTrackCanvas(QWidget):
    """Custom-painted waveform track with label overlay, playhead cursor,
    and played-portion highlight. No matplotlib dependency."""

    seek_requested = pyqtSignal(float)

    def __init__(self, label: str, audio_data: np.ndarray, sample_rate: int,
                 colors: dict, parent=None):
        super().__init__(parent)
        self._label = label
        self._colors = colors
        self._sample_rate = sample_rate
        self._muted = False
        self._playhead_frac = 0.0  # 0.0 to 1.0
        self._dragging = False

        self._set_waveform(audio_data)

        self.setMinimumHeight(50)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _set_waveform(self, audio_data: np.ndarray):
        # Downsample waveform for display
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)
        self._duration = len(audio_data) / self._sample_rate if self._sample_rate else 0.0

        max_points = 2000
        if len(audio_data) > max_points:
            step = len(audio_data) // max_points
            self._wave = audio_data[::step].copy()
        else:
            self._wave = audio_data.copy()

        # Normalize
        peak = np.max(np.abs(self._wave))
        if peak > 0:
            self._wave /= peak

    def set_audio_data(self, audio_data: np.ndarray):
        self._set_waveform(audio_data)
        self.update()

    def set_muted(self, muted: bool):
        self._muted = muted
        self.update()

    def set_playhead(self, fraction: float):
        """Set playhead position as fraction 0.0-1.0."""
        self._playhead_frac = max(0.0, min(1.0, fraction))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        border_color = QColor(self._colors["border"])
        fill_color = QColor(self._colors["fill"])
        wave_color = QColor(self._colors["wave"])

        # Background with rounded rect and border
        p.setPen(QPen(border_color, 2))
        p.setBrush(fill_color)
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 6, 6)

        # Played portion highlight
        if self._playhead_frac > 0:
            played_w = self._playhead_frac * (w - 4)
            highlight = QColor(self._colors["wave"])
            highlight.setAlpha(25)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(highlight)
            p.drawRoundedRect(QRectF(2, 2, played_w, h - 4), 5, 5)

        # Draw waveform
        if not self._muted and len(self._wave) > 0:
            margin_x = 8
            margin_y = 12
            draw_w = w - 2 * margin_x
            draw_h = (h - 2 * margin_y) / 2
            cy = h / 2

            wave_color_obj = QColor(wave_color)
            wave_color_obj.setAlpha(200)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(wave_color_obj)

            n = len(self._wave)
            bar_w = max(1, draw_w / n)

            for i in range(n):
                x = margin_x + (i / n) * draw_w
                amp = abs(self._wave[i]) * draw_h
                if amp < 0.5:
                    amp = 0.5
                p.drawRect(QRectF(x, cy - amp, bar_w, amp * 2))

        # Label overlay
        p.setPen(QColor("#e0e0e0"))
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(QRectF(12, 4, w - 24, h - 8),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                   self._label)

        # Playhead cursor line
        if self._playhead_frac > 0:
            px = 2 + self._playhead_frac * (w - 4)
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.drawLine(QPointF(px, 2), QPointF(px, h - 2))

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._emit_seek(event.position().x())

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._emit_seek(event.position().x())

    def _emit_seek(self, x: float):
        frac = max(0.0, min(1.0, (x - 2) / max(1, self.width() - 4)))
        t = frac * self._duration
        self.seek_requested.emit(t)


class StemTrackWidget(QWidget):
    """A single horizontal track row matching the reference design."""

    mute_changed = pyqtSignal(int, bool)       # track_index, muted
    seek_requested = pyqtSignal(float)         # time in seconds

    def __init__(self, track_index: int, name: str, audio_data: np.ndarray,
                 sample_rate: int, color_index: int = 0, initially_muted: bool = False,
                 parent=None):
        super().__init__(parent)
        self._index = track_index
        self._name = name
        self._sample_rate = sample_rate
        self._duration = len(audio_data) / sample_rate if audio_data.ndim == 1 else audio_data.shape[0] / sample_rate
        self._colors = get_track_color(color_index)

        self.setFixedHeight(70)
        self.setStyleSheet("StemTrackWidget { background: transparent; }")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        # Speaker/Mute button
        self._mute_btn = _SpeakerButton()
        self._mute_btn.setStyleSheet(
            "QPushButton { background-color: #2a2a3c; border: 1px solid #45475a; border-radius: 20px; }"
            "QPushButton:hover { background-color: #3a3a4c; }"
            "QPushButton:checked { background-color: #2a2a3c; }"
        )
        self._mute_btn.clicked.connect(self._on_mute)
        layout.addWidget(self._mute_btn)

        # Waveform canvas with label
        self._canvas = WaveformTrackCanvas(name, audio_data, sample_rate, self._colors)
        self._canvas.seek_requested.connect(self.seek_requested)
        layout.addWidget(self._canvas, stretch=1)

        # Apply initial mute state
        if initially_muted:
            self._mute_btn.set_muted(True)
            self._canvas.set_muted(True)

    def update_playhead(self, time_seconds: float):
        if self._duration > 0:
            frac = time_seconds / self._duration
            self._canvas.set_playhead(frac)

    def set_muted(self, muted: bool):
        self._mute_btn.set_muted(muted)
        self._canvas.set_muted(muted)

    def set_audio_data(self, audio_data: np.ndarray):
        self._duration = len(audio_data) / self._sample_rate if self._sample_rate else 0.0
        self._canvas.set_audio_data(audio_data)

    def _on_mute(self):
        muted = self._mute_btn.isChecked()
        self._mute_btn.set_muted(muted)
        self._canvas.set_muted(muted)
        self.mute_changed.emit(self._index, muted)
