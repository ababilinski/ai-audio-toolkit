"""Audio analysis utilities: BPM and key detection using librosa."""
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class AudioAnalysisWorker(QThread):
    """Detect BPM and musical key in a background thread."""
    finished = pyqtSignal(float, str)  # bpm, key_string
    error = pyqtSignal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        try:
            import librosa

            y, sr = librosa.load(self.file_path, sr=22050, mono=True)

            # BPM detection
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

            # Key detection via Krumhansl-Schmuckler
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_avg = chroma.mean(axis=1)

            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                         'F#', 'G', 'G#', 'A', 'A#', 'B']
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

            best_corr = -1.0
            best_key = "C MAJOR"
            for shift in range(12):
                rolled = np.roll(chroma_avg, -shift)
                maj_corr = float(np.corrcoef(rolled, major_profile)[0, 1])
                min_corr = float(np.corrcoef(rolled, minor_profile)[0, 1])
                if maj_corr > best_corr:
                    best_corr = maj_corr
                    best_key = f"{key_names[shift]} MAJOR"
                if min_corr > best_corr:
                    best_corr = min_corr
                    best_key = f"{key_names[shift]} MINOR"

            self.finished.emit(bpm, best_key)
        except Exception as e:
            self.error.emit(str(e))
