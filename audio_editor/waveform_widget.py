"""Waveform visualization widget using matplotlib embedded in PyQt6."""
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class WaveformWidget(QWidget):
    """Displays audio waveform with click/drag to seek."""

    seek_requested = pyqtSignal(float)  # time in seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 2), dpi=100)
        self.figure.patch.set_facecolor("#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._ax = self.figure.add_subplot(111)
        self._playhead = None
        self._played_region = None
        self._duration = 0.0
        self._dragging = False
        self._style_axis()
        self.canvas.draw()

        # Connect mouse events for scrubbing
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _style_axis(self):
        self._ax.set_facecolor("#1e1e2e")
        self._ax.tick_params(colors="#888888", labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_color("#333344")
        self._ax.set_xlabel("")
        self._ax.set_ylabel("")

    def plot_waveform(self, audio_data: np.ndarray, sample_rate: int, title: str = ""):
        self._ax.clear()
        self._playhead = None
        self._played_region = None
        self._style_axis()

        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)

        self._duration = len(audio_data) / sample_rate

        # Downsample for display performance
        max_points = 10000
        if len(audio_data) > max_points:
            step = len(audio_data) // max_points
            audio_data = audio_data[::step]
            times = np.arange(len(audio_data)) * step / sample_rate
        else:
            times = np.arange(len(audio_data)) / sample_rate

        self._ax.fill_between(times, audio_data, alpha=0.6, color="#89b4fa")
        self._ax.fill_between(times, -np.abs(audio_data), alpha=0.6, color="#89b4fa")
        self._ax.set_xlim(times[0], times[-1])

        max_val = np.max(np.abs(audio_data)) * 1.1
        if max_val > 0:
            self._ax.set_ylim(-max_val, max_val)

        if title:
            self._ax.set_title(title, color="#cdd6f4", fontsize=9, pad=4)

        self._ax.set_xlabel("Time (s)", color="#888888", fontsize=7)
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()

    def update_playhead(self, time_seconds: float):
        # Played-portion highlight
        if self._played_region:
            self._played_region.remove()
            self._played_region = None
        if time_seconds > 0 and self._duration > 0:
            self._played_region = self._ax.axvspan(
                0, time_seconds, alpha=0.08, color="#89b4fa"
            )
        # Playhead cursor line
        if self._playhead:
            self._playhead.set_xdata([time_seconds, time_seconds])
        else:
            self._playhead = self._ax.axvline(
                x=time_seconds, color="#ffffff", linewidth=1.5, alpha=0.9
            )
        self.canvas.draw_idle()

    def clear(self):
        self._ax.clear()
        self._playhead = None
        self._played_region = None
        self._duration = 0.0
        self._style_axis()
        self.canvas.draw()

    # ── Mouse scrubbing ──

    def _on_press(self, event):
        if event.inaxes == self._ax and event.button == 1:
            self._dragging = True
            self._seek_to(event.xdata)

    def _on_release(self, event):
        self._dragging = False

    def _on_motion(self, event):
        if self._dragging and event.inaxes == self._ax and event.xdata is not None:
            self._seek_to(event.xdata)

    def _seek_to(self, x: float):
        if x is None or self._duration <= 0:
            return
        t = max(0.0, min(x, self._duration))
        self.update_playhead(t)
        self.seek_requested.emit(t)
