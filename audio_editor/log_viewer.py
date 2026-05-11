"""Console log viewer widget that captures Python logging and stdout/stderr."""
import io
import logging
import sys
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QCheckBox,
)
from PyQt6.QtGui import QTextCursor, QColor


class LogSignalEmitter(QObject):
    """Bridge to emit log records as Qt signals (thread-safe)."""
    log_received = pyqtSignal(str, int)  # message, level


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal."""

    def __init__(self, emitter: LogSignalEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_received.emit(msg, record.levelno)


class _StreamCapture(io.TextIOBase):
    """Captures writes to stdout/stderr and forwards to the log viewer."""

    def __init__(self, emitter: LogSignalEmitter, level: int, original):
        super().__init__()
        self._emitter = emitter
        self._level = level
        self._original = original

    def write(self, text):
        if self._original:
            self._original.write(text)
        text = text.rstrip("\n\r")
        if text:
            self._emitter.log_received.emit(text, self._level)
        return len(text)

    def flush(self):
        if self._original:
            self._original.flush()

    def fileno(self):
        if self._original:
            return self._original.fileno()
        raise io.UnsupportedOperation("fileno")

    @property
    def encoding(self):
        if self._original:
            return self._original.encoding
        return "utf-8"


LEVEL_COLORS = {
    logging.DEBUG: "#6c7086",
    logging.INFO: "#a6adc8",
    logging.WARNING: "#f9e2af",
    logging.ERROR: "#f38ba8",
    logging.CRITICAL: "#f38ba8",
}


class LogViewerWidget(QWidget):
    """Scrollable log viewer panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._autoscroll_cb = QCheckBox("Auto-scroll")
        self._autoscroll_cb.setChecked(True)
        toolbar.addWidget(self._autoscroll_cb)
        toolbar.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)

        # Log text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(
            "QTextEdit { background-color: #11111b; color: #a6adc8; "
            "font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 11px; "
            "border: 1px solid #333344; border-radius: 4px; padding: 4px; }"
        )
        layout.addWidget(self._text)

        # Set up log capture
        self._emitter = LogSignalEmitter()
        self._emitter.log_received.connect(self._append_log)
        self._handler = QtLogHandler(self._emitter)
        self._handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

        self._orig_stdout = None
        self._orig_stderr = None

    def install(self):
        """Install the log handler on the root logger and capture stdout/stderr."""
        root = logging.getLogger()
        root.addHandler(self._handler)
        root.setLevel(logging.DEBUG)

        # Capture stdout/stderr so print() and third-party output appears in console
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _StreamCapture(self._emitter, logging.INFO, self._orig_stdout)
        sys.stderr = _StreamCapture(self._emitter, logging.WARNING, self._orig_stderr)

    def uninstall(self):
        logging.getLogger().removeHandler(self._handler)
        if self._orig_stdout:
            sys.stdout = self._orig_stdout
        if self._orig_stderr:
            sys.stderr = self._orig_stderr

    def _append_log(self, message: str, level: int):
        color = LEVEL_COLORS.get(level, "#a6adc8")
        # Escape HTML to avoid rendering issues with log messages
        import html
        safe = html.escape(message)
        self._text.append(f'<span style="color: {color};">{safe}</span>')
        if self._autoscroll_cb.isChecked():
            self._text.moveCursor(QTextCursor.MoveOperation.End)

    def _clear(self):
        self._text.clear()
