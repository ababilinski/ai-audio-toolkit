"""Background worker for live effects preview rendering."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
import threading

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .effects import apply_effect_stack, normalize_effect_state


class LiveEffectsPreviewProcessor(QObject):
    """Process the latest requested layer previews off the UI thread."""

    result_ready = pyqtSignal(int, str, int, object, object)
    error = pyqtSignal(int, str, int, str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending: OrderedDict[str, tuple[int, str, int, np.ndarray, int, dict[str, dict[str, float | bool]]]] = OrderedDict()
        self._running = True
        self._busy = False
        self._thread = threading.Thread(target=self._run, name="effects-preview-worker", daemon=True)
        self._thread.start()

    def submit(
        self,
        session_id: int,
        layer_key: str,
        request_id: int,
        audio_data: np.ndarray,
        sample_rate: int,
        state: dict[str, dict[str, float | bool]],
    ) -> None:
        normalized = normalize_effect_state(state)
        audio = np.asarray(audio_data, dtype=np.float32)
        with self._condition:
            if layer_key in self._pending:
                self._pending.pop(layer_key)
            self._pending[layer_key] = (
                session_id,
                layer_key,
                request_id,
                audio,
                int(sample_rate),
                normalized,
            )
            if not self._busy:
                self._busy = True
                self.busy_changed.emit(True)
            self._condition.notify()

    def clear_pending(self) -> None:
        with self._condition:
            self._pending.clear()

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and not self._pending:
                    self._condition.wait()
                if not self._running:
                    return
                session_id, layer_key, request_id, audio, sample_rate, state = self._pending.popitem(last=False)[1]

            try:
                processed = apply_effect_stack(audio, sample_rate, state)
            except Exception as exc:  # pragma: no cover - defensive cross-thread path
                self.error.emit(session_id, layer_key, request_id, str(exc))
            else:
                self.result_ready.emit(session_id, layer_key, request_id, processed, state)

            with self._condition:
                if not self._pending:
                    self._busy = False
                    self.busy_changed.emit(False)
