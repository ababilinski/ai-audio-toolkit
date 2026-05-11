"""Multi-track audio engine with per-stem volume, mute, solo, and scrubbing."""
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from dataclasses import dataclass, field
from pathlib import Path
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot, QMetaObject, Qt


@dataclass
class StemTrack:
    """A single stem loaded into memory."""
    name: str
    file_path: str
    data: np.ndarray  # shape: (samples, channels)
    sample_rate: int
    source_data: np.ndarray | None = None
    volume: float = 1.0
    muted: bool = False
    solo: bool = False


class AudioEngine(QObject):
    """Manages multi-track playback with mixing, volume, mute/solo, and seeking."""

    position_changed = pyqtSignal(float)  # current time in seconds
    playback_finished = pyqtSignal()
    playback_started = pyqtSignal()
    playback_paused = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: list[StemTrack] = []
        self._original: StemTrack | None = None
        self._sample_rate: int = 44100
        self._frame_pos: int = 0
        self._total_frames: int = 0
        self._playing: bool = False
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

        # Timer to emit position updates
        self._timer = QTimer(self)
        self._timer.setInterval(50)  # 20 fps updates
        self._timer.timeout.connect(self._emit_position)

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def duration(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return self._total_frames / self._sample_rate

    @property
    def position(self) -> float:
        return self._frame_pos / self._sample_rate

    @property
    def tracks(self) -> list[StemTrack]:
        return self._tracks

    @property
    def original(self) -> StemTrack | None:
        return self._original

    def load_original(self, file_path: str):
        """Load the original (unseparated) audio file."""
        data, sr = sf.read(file_path, dtype="float32", always_2d=True)
        self._original = StemTrack(
            name="Original",
            file_path=file_path,
            data=data,
            sample_rate=sr,
            source_data=data.copy(),
        )
        self._sample_rate = sr
        self._total_frames = max(self._total_frames, len(data))

    def load_stems(self, file_paths: list[str]):
        """Load separated stem files."""
        self._tracks.clear()
        max_len = 0
        for path in file_paths:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            # Resample if needed to match
            if sr != self._sample_rate and self._sample_rate > 0:
                # Simple nearest-neighbor resample (good enough for playback)
                ratio = self._sample_rate / sr
                new_len = int(len(data) * ratio)
                indices = np.clip((np.arange(new_len) / ratio).astype(int), 0, len(data) - 1)
                data = data[indices]
                sr = self._sample_rate

            name = self._display_name_for_stem(path)

            self._tracks.append(StemTrack(
                name=name,
                file_path=path,
                data=data,
                sample_rate=sr,
                source_data=data.copy(),
            ))
            max_len = max(max_len, len(data))

        self._sample_rate = self._tracks[0].sample_rate if self._tracks else 44100
        self._total_frames = max_len
        self._frame_pos = 0

    @staticmethod
    def _display_name_for_stem(file_path: str) -> str:
        """Map generated stem filenames to cleaner labels for the UI."""
        stem = Path(file_path).stem
        normalized = stem.lower().replace("-", "_").replace(" ", "_")

        named_patterns = [
            ("speaker_voice", "Speaker / voice"),
            ("background_bleed_room", "Background / bleed / room"),
            ("noreverb", "No reverb"),
            ("no_reverb", "No reverb"),
            ("dereverb", "No reverb"),
            ("reverb", "Reverb / room"),
            ("dry", "Dry / direct"),
        ]
        for token, label in named_patterns:
            if token in normalized:
                return label

        for token, label in [
            ("(vocals)", "Vocals"),
            ("(instrumental)", "Instrumental"),
            ("(no vocals)", "No vocals"),
            ("(other)", "Other"),
        ]:
            if token in stem.lower():
                return label

        return stem

    def clear(self):
        """Remove all tracks."""
        self.stop()
        self._tracks.clear()
        self._original = None
        self._total_frames = 0
        self._frame_pos = 0

    def set_volume(self, track_index: int, volume: float):
        """Set volume for a track (0.0 to 1.0)."""
        if 0 <= track_index < len(self._tracks):
            self._tracks[track_index].volume = max(0.0, min(1.0, volume))

    def set_muted(self, track_index: int, muted: bool):
        if 0 <= track_index < len(self._tracks):
            self._tracks[track_index].muted = muted

    def set_solo(self, track_index: int, solo: bool):
        if 0 <= track_index < len(self._tracks):
            self._tracks[track_index].solo = solo

    def set_original_volume(self, volume: float):
        if self._original:
            self._original.volume = max(0.0, min(1.0, volume))

    def set_original_muted(self, muted: bool):
        if self._original:
            self._original.muted = muted

    def update_track_audio(self, track_index: int, audio_data: np.ndarray):
        """Replace a track's playback buffer without losing its original source data."""
        if 0 <= track_index < len(self._tracks):
            with self._lock:
                self._tracks[track_index].data = np.asarray(audio_data, dtype=np.float32)
                self._total_frames = max(self._total_frames, len(self._tracks[track_index].data))

    def update_original_audio(self, audio_data: np.ndarray):
        """Replace the original track's playback buffer."""
        if self._original is None:
            return
        with self._lock:
            self._original.data = np.asarray(audio_data, dtype=np.float32)
            self._total_frames = max(self._total_frames, len(self._original.data))

    def seek(self, time_seconds: float):
        """Seek to a position in seconds."""
        with self._lock:
            self._frame_pos = int(time_seconds * self._sample_rate)
            self._frame_pos = max(0, min(self._frame_pos, self._total_frames))

    def play(self):
        """Start or resume playback."""
        if self._playing:
            return
        if not self._tracks and not self._original:
            return

        self._playing = True

        channels = 2
        if self._tracks:
            channels = self._tracks[0].data.shape[1]
        elif self._original:
            channels = self._original.data.shape[1]

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=channels,
            dtype="float32",
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
            blocksize=2048,
        )
        self._stream.start()
        self._timer.start()
        self.playback_started.emit()

    def pause(self):
        """Pause playback."""
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        if self._stream:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.playback_paused.emit()

    def stop(self):
        """Stop playback and reset position."""
        self._playing = False
        self._timer.stop()
        if self._stream:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._frame_pos = 0
        self.playback_finished.emit()

    def play_single(self, file_path: str):
        """Play a single file (for previewing original)."""
        self.stop()
        data, sr = sf.read(file_path, dtype="float32", always_2d=True)
        self._original = StemTrack("Preview", file_path, data, sr)
        self._tracks.clear()
        self._sample_rate = sr
        self._total_frames = len(data)
        self._frame_pos = 0
        self._original.muted = False
        self._original.volume = 1.0
        self.play()

    def play_single_stem(self, track_index: int):
        """Play only one stem track in isolation (solo it internally)."""
        self.pause()
        # Reset all solo flags
        for t in self._tracks:
            t.solo = False
        if self._original:
            self._original.solo = False
        # Solo the requested track
        if 0 <= track_index < len(self._tracks):
            self._tracks[track_index].solo = True
        elif track_index == -1 and self._original:
            self._original.solo = True
        self.play()

    def play_preview(self, audio_data: np.ndarray, sample_rate: int):
        """Play arbitrary audio data (used for effects preview)."""
        self.stop()
        self._original = StemTrack("Preview", "", audio_data, sample_rate)
        self._original.muted = False
        self._original.volume = 1.0
        saved_tracks = self._tracks
        self._tracks = []
        self._sample_rate = sample_rate
        self._total_frames = len(audio_data)
        self._frame_pos = 0
        self.play()
        # Restore tracks after starting (engine references saved list)
        # Note: tracks restored when stop/pause clears state
        self._tracks = saved_tracks

    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback — runs in audio thread."""
        with self._lock:
            start = self._frame_pos
            end = start + frames

            if start >= self._total_frames:
                outdata[:] = 0
                raise sd.CallbackStop()

            actual_end = min(end, self._total_frames)
            actual_frames = actual_end - start
            out_channels = outdata.shape[1]

            mixed = np.zeros((frames, out_channels), dtype=np.float32)

            # Determine which tracks to include
            any_solo = any(t.solo for t in self._tracks)
            if self._original and self._original.solo:
                any_solo = True

            # Mix tracks
            all_sources = []
            if self._original:
                all_sources.append(self._original)
            all_sources.extend(self._tracks)

            for track in all_sources:
                if track.muted:
                    continue
                if any_solo and not track.solo:
                    continue

                if start < len(track.data):
                    t_end = min(actual_end, len(track.data))
                    t_frames = t_end - start
                    chunk = track.data[start:t_end] * track.volume

                    # Handle channel mismatch
                    if chunk.shape[1] < out_channels:
                        chunk = np.repeat(chunk, out_channels // chunk.shape[1], axis=1)
                    elif chunk.shape[1] > out_channels:
                        chunk = chunk[:, :out_channels]

                    mixed[:t_frames] += chunk

            # Clip to prevent distortion
            np.clip(mixed, -1.0, 1.0, out=mixed)
            outdata[:] = mixed
            self._frame_pos = actual_end

    def _on_stream_finished(self):
        """Called from the audio thread — marshal to main thread."""
        self._playing = False
        # QTimer.stop() must be called from the main thread
        QMetaObject.invokeMethod(self._timer, "stop", Qt.ConnectionType.QueuedConnection)
        QMetaObject.invokeMethod(self, "_emit_finished", Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _emit_finished(self):
        self.playback_finished.emit()

    def _emit_position(self):
        self.position_changed.emit(self.position)
