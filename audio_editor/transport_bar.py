"""Transport bar: Play/Pause, Stop, time display, Split Another Track."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QColor, QPolygonF, QFont
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QSizePolicy


def _format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class _PlayPauseButton(QPushButton):
    """Custom-painted play/pause button for reliable cross-platform rendering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playing = False
        self.setFixedSize(40, 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_playing(self, playing: bool):
        self._playing = playing
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#1e1e2e")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)

        cx, cy = self.width() / 2, self.height() / 2

        if self._playing:
            # Draw pause bars
            bar_w, bar_h = 4, 14
            gap = 4
            p.drawRoundedRect(QRectF(cx - gap - bar_w, cy - bar_h / 2, bar_w, bar_h), 1, 1)
            p.drawRoundedRect(QRectF(cx + gap, cy - bar_h / 2, bar_w, bar_h), 1, 1)
        else:
            # Draw play triangle
            tri = QPolygonF([
                QPointF(cx - 5, cy - 8),
                QPointF(cx - 5, cy + 8),
                QPointF(cx + 7, cy),
            ])
            p.drawPolygon(tri)
        p.end()


class _StopButton(QPushButton):
    """Custom-painted stop button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#cdd6f4"))
        cx, cy = self.width() / 2, self.height() / 2
        size = 12
        p.drawRoundedRect(QRectF(cx - size / 2, cy - size / 2, size, size), 2, 2)
        p.end()


class TransportBar(QWidget):
    """Play/Pause, Stop, and time display. No seek slider (scrubbing is on waveforms)."""

    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    split_another_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration = 0.0
        self._is_playing = False

        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Play/Pause
        self._play_btn = _PlayPauseButton()
        self._play_btn.setStyleSheet(
            "QPushButton { background-color: #89b4fa; border: none; border-radius: 20px; }"
            "QPushButton:hover { background-color: #b4d0fb; }"
            "QPushButton:pressed { background-color: #74a8f7; }"
        )
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._on_play_pause)
        layout.addWidget(self._play_btn)

        # Stop
        self._stop_btn = _StopButton()
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 6px; }"
            "QPushButton:hover { background-color: #45475a; }"
            "QPushButton:pressed { background-color: #585b70; }"
        )
        self._stop_btn.setToolTip("Stop")
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self._stop_btn)

        # Time display
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setMinimumWidth(110)
        self._time_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._time_label.setStyleSheet(
            "color: #cdd6f4; font-size: 14px; font-family: 'Consolas', 'Courier New', monospace;"
        )
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._time_label)

        layout.addStretch()

        # Split Another Track
        self._split_btn = QPushButton("Split Another Track")
        self._split_btn.setStyleSheet(
            "QPushButton { background-color: #313244; color: #cdd6f4; "
            "border: 1px solid #45475a; border-radius: 6px; padding: 8px 18px; font-size: 12px; }"
            "QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }"
        )
        self._split_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._split_btn.setMinimumWidth(max(150, self._split_btn.sizeHint().width()))
        self._split_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._split_btn.clicked.connect(self.split_another_clicked.emit)
        layout.addWidget(self._split_btn)

    def set_duration(self, seconds: float):
        self._duration = seconds
        self._update_time_label(0.0)

    def set_position(self, seconds: float):
        self._update_time_label(seconds)

    def set_playing(self, playing: bool):
        self._is_playing = playing
        self._play_btn.set_playing(playing)

    def _update_time_label(self, current: float):
        self._time_label.setText(f"{_format_time(current)} / {_format_time(self._duration)}")

    def _on_play_pause(self):
        if self._is_playing:
            self.pause_clicked.emit()
        else:
            self.play_clicked.emit()
