"""Backend wrapper for audio-separator with organized model presets."""
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal


def _separator_model_dir() -> str:
    """Resolve the managed audio-separator model directory."""
    from .settings import get_app_settings

    path = Path(get_app_settings().separator_model_dir())
    path.mkdir(parents=True, exist_ok=True)
    os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = str(path)
    return str(path)


@dataclass
class ModelPreset:
    """A named model preset with metadata."""
    name: str
    model_filename: str
    description: str
    category: str
    stems: list[str] = field(default_factory=list)
    architecture: str = ""


# Best publicly available models organized by use case
MODEL_PRESETS: dict[str, list[ModelPreset]] = {
    "Vocal Separation": [
        ModelPreset(
            name="BS-Roformer (Best Quality)",
            model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            description=(
                "State-of-the-art vocal separation using Band-Split Roformer transformer.\n"
                "Best for: studio recordings, clean music mixes, any genre.\n"
                "Vocals stem = lead + backing vocals. Instrumental = everything else.\n"
                "SDR 12.97 — highest of any public vocal model. Start here for best results.\n"
                "Tip: Overlap 8–12 for cleaner boundaries on long songs."
            ),
            category="Vocal Separation",
            stems=["vocals", "instrumental"],
            architecture="Roformer",
        ),
        ModelPreset(
            name="Kim Vocal 2",
            model_filename="Kim_Vocal_2.onnx",
            description=(
                "Lightweight MDX-Net ONNX model specifically tuned for vocal isolation.\n"
                "Best for: quick extraction, real-time workflows, batch processing.\n"
                "Vocals stem = lead + backing vocals. Instrumental = everything else.\n"
                "Faster than Roformer with slightly lower SDR. ONNX = excellent RTX GPU use.\n"
                "Tip: Good choice when speed matters more than maximum quality."
            ),
            category="Vocal Separation",
            stems=["vocals", "instrumental"],
            architecture="MDX-Net",
        ),
        ModelPreset(
            name="InstVoc HQ (MDX23C)",
            model_filename="MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
            description=(
                "High-quality vocal/instrumental split using the MDX23C architecture.\n"
                "Best for: clean studio recordings, pop and rock music.\n"
                "Vocals stem = lead + backing vocals. Instrumental = full band minus vocals.\n"
                "8K FFT size gives finer frequency resolution than standard MDX models.\n"
                "Tip: Overlap 8–12 recommended. Use Segment 256 for 8GB VRAM."
            ),
            category="Vocal Separation",
            stems=["vocals", "instrumental"],
            architecture="MDXC",
        ),
        ModelPreset(
            name="UVR MDX-NET Inst HQ 4",
            model_filename="UVR-MDX-NET-Inst_HQ_4.onnx",
            description=(
                "High-quality ONNX model focused on instrumental extraction.\n"
                "Best for: when you want a clean instrumental track (music minus vocals).\n"
                "Vocals stem = extracted vocals (may have some bleed). Instrumental = primary output.\n"
                "ONNX format means fast GPU acceleration on any NVIDIA GPU.\n"
                "Tip: Enable denoise pass for recordings with background noise."
            ),
            category="Vocal Separation",
            stems=["vocals", "instrumental"],
            architecture="MDX-Net",
        ),
        ModelPreset(
            name="Kuielab Vocals",
            model_filename="kuielab_a_vocals.onnx",
            description=(
                "MDX-Net ONNX model from the Kuielab series, trained for vocal isolation.\n"
                "Best for: general vocal separation, fast processing pipelines.\n"
                "Vocals stem = lead + backing vocals. Other = remaining instruments.\n"
                "Lighter model — lower VRAM requirement, good for weaker GPUs.\n"
                "Tip: Pair with a noise removal model if source is noisy."
            ),
            category="Vocal Separation",
            stems=["vocals", "other"],
            architecture="MDX-Net",
        ),
        ModelPreset(
            name="4-HP Vocal UVR",
            model_filename="4_HP-Vocal-UVR.pth",
            description=(
                "VR (Volume Regression) architecture model for vocal separation.\n"
                "Best for: general vocal extraction, older recordings, mixed quality sources.\n"
                "Vocals stem = lead + backing vocals. Instrumental = everything else.\n"
                "VR models use different processing than MDX/Roformer — can handle edge cases.\n"
                "Tip: Window 320 gives best quality; TTA checkbox for maximum quality."
            ),
            category="Vocal Separation",
            stems=["vocals", "instrumental"],
            architecture="VR",
        ),
    ],
    "Karaoke": [
        ModelPreset(
            name="MelBand Roformer Karaoke (Best)",
            model_filename="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            description=(
                "Best karaoke model — uses mel-band Roformer to isolate lead vocals only.\n"
                "Best for: creating karaoke tracks, removing lead singer while keeping backing vocals.\n"
                "Vocals stem = lead vocals only. Karaoke stem = music + all backing vocals.\n"
                "SDR 10.19 — specifically trained on lead vocal isolation from full arrangements.\n"
                "Tip: Overlap 8–16 for cleaner lead vocal boundaries on long songs."
            ),
            category="Karaoke",
            stems=["vocals", "karaoke"],
            architecture="Roformer",
        ),
        ModelPreset(
            name="UVR MDX-NET Karaoke 2",
            model_filename="UVR_MDXNET_KARA_2.onnx",
            description=(
                "Fast ONNX karaoke model for quick lead vocal removal.\n"
                "Best for: batch karaoke creation, real-time preview, GPU-accelerated workflows.\n"
                "Vocals stem = lead vocals. Karaoke stem = music + backing vocals.\n"
                "Second generation of the UVR karaoke series — improved over original.\n"
                "Tip: Good balance of speed and quality for most pop/rock songs."
            ),
            category="Karaoke",
            stems=["vocals", "karaoke"],
            architecture="MDX-Net",
        ),
        ModelPreset(
            name="6-HP Karaoke UVR",
            model_filename="6_HP-Karaoke-UVR.pth",
            description=(
                "6-layer VR architecture karaoke model.\n"
                "Best for: songs where other karaoke models leave too much lead vocal.\n"
                "Vocals stem = lead vocals. Karaoke stem = music + backing vocals.\n"
                "More aggressive vocal removal than 4-HP — may affect some backing vocals.\n"
                "Tip: Try Aggression 4–5; Window 320 for best clarity."
            ),
            category="Karaoke",
            stems=["vocals", "karaoke"],
            architecture="VR",
        ),
    ],
    "Multi-Stem (Full Band)": [
        ModelPreset(
            name="HTDemucs Fine-Tuned (4 stems)",
            model_filename="htdemucs_ft.yaml",
            description=(
                "Facebook's best Demucs model — fine-tuned separately per stem for maximum quality.\n"
                "Best for: full music production, stem mastering, remixing any genre.\n"
                "Stems: vocals, drums, bass, other (guitars, synths, keys, etc.).\n"
                "Fine-tuned = each stem has its own specialised model — noticeably better than Standard.\n"
                "Tip: Shifts 2–5 for best quality. Use High Quality preset for final exports."
            ),
            category="Multi-Stem (Full Band)",
            stems=["vocals", "drums", "bass", "other"],
            architecture="Demucs",
        ),
        ModelPreset(
            name="HTDemucs 6-Stem",
            model_filename="htdemucs_6s.yaml",
            description=(
                "6-stem Demucs model that separates guitar and piano as individual stems.\n"
                "Best for: music with clear guitar or piano parts you want isolated.\n"
                "Stems: vocals, drums, bass, guitar, piano, other.\n"
                "Guitar and piano stems require those instruments to be prominent in the mix.\n"
                "Tip: Works best on recordings where guitar/piano are clearly audible. "
                "Weak on faint parts."
            ),
            category="Multi-Stem (Full Band)",
            stems=["vocals", "drums", "bass", "guitar", "piano", "other"],
            architecture="Demucs",
        ),
        ModelPreset(
            name="HTDemucs Standard",
            model_filename="htdemucs.yaml",
            description=(
                "The standard HTDemucs model — good all-around 4-stem separation.\n"
                "Best for: general use when Fine-Tuned is overkill or too slow.\n"
                "Stems: vocals, drums, bass, other (guitars, synths, etc.).\n"
                "Faster than Fine-Tuned but slightly lower quality per stem.\n"
                "Tip: Use Fine-Tuned instead if quality matters; this is for speed."
            ),
            category="Multi-Stem (Full Band)",
            stems=["vocals", "drums", "bass", "other"],
            architecture="Demucs",
        ),
        ModelPreset(
            name="HDemucs MMI",
            model_filename="hdemucs_mmi.yaml",
            description=(
                "Hybrid Demucs trained on Musdb18-HQ + extra mixed media (MMI) data.\n"
                "Best for: diverse or unusual audio sources, non-Western music, TV audio.\n"
                "Stems: vocals, drums, bass, other.\n"
                "MMI training data includes speech, sound effects, and varied music styles.\n"
                "Tip: Try this when standard HTDemucs gives poor results on unusual content."
            ),
            category="Multi-Stem (Full Band)",
            stems=["vocals", "drums", "bass", "other"],
            architecture="Demucs",
        ),
    ],
    "Individual Instruments": [
        ModelPreset(
            name="Kuielab Bass",
            model_filename="kuielab_a_bass.onnx",
            description=(
                "MDX-Net ONNX model specifically trained to isolate bass guitar/bass frequencies.\n"
                "Best for: extracting bass lines from full mixes, bass transcription.\n"
                "Bass stem = bass guitar, sub-bass, bass synth. Other = everything else.\n"
                "Low frequencies are hard to separate — expect some bleed from kick drum.\n"
                "Tip: Works best on mixes where bass is prominent and clearly panned centre."
            ),
            category="Individual Instruments",
            stems=["bass", "other"],
            architecture="MDX-Net",
        ),
        ModelPreset(
            name="Kuielab Drums",
            model_filename="kuielab_a_drums.onnx",
            description=(
                "MDX-Net ONNX model specifically trained to isolate drum/percussion tracks.\n"
                "Best for: drum transcription, drum stem extraction, rhythm analysis.\n"
                "Drums stem = kick, snare, hi-hats, cymbals, toms. Other = everything else.\n"
                "Transient-heavy drums separate cleanly — generally high quality.\n"
                "Tip: Enable denoise pass if the extracted drums have hiss or noise."
            ),
            category="Individual Instruments",
            stems=["drums", "other"],
            architecture="MDX-Net",
        ),
        ModelPreset(
            name="Kuielab Other",
            model_filename="kuielab_a_other.onnx",
            description=(
                "MDX-Net ONNX model that extracts 'other' instruments (guitars, keys, synths).\n"
                "Best for: isolating melodic instruments that aren't vocals, drums, or bass.\n"
                "Other stem = guitars, piano, strings, synths, etc. Vocals stem = extracted vocals.\n"
                "Designed as the complement to Kuielab Bass and Drums in a full pipeline.\n"
                "Tip: Use all three Kuielab models together for a full 4-stem split."
            ),
            category="Individual Instruments",
            stems=["other", "vocals"],
            architecture="MDX-Net",
        ),
    ],
    "Audio Cleanup": [
        ModelPreset(
            name="Denoise MelBand Roformer (Best)",
            model_filename="denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            description=(
                "Best noise removal model — mel-band Roformer trained specifically for denoising.\n"
                "Best for: removing hiss, hum, fan noise, air conditioning, background chatter.\n"
                "Clean stem = denoised audio. Noise stem = extracted noise (useful for checking).\n"
                "SDR 27.99 — highest quality noise separation available publicly.\n"
                "Tip: Overlap 12–16 for smoother noise floor. Works on speech and music."
            ),
            category="Audio Cleanup",
            stems=["clean", "noise"],
            architecture="Roformer",
        ),
        ModelPreset(
            name="UVR DeNoise",
            model_filename="UVR-DeNoise.pth",
            description=(
                "General-purpose VR architecture noise removal model.\n"
                "Best for: lighter noise reduction, preserving more natural room sound.\n"
                "Clean stem = denoised audio. Noise stem = extracted noise.\n"
                "Lighter model — faster than Roformer, slightly less aggressive.\n"
                "Tip: Try this first for gentle cleanup; use MelBand Roformer for heavy noise."
            ),
            category="Audio Cleanup",
            stems=["clean", "noise"],
            architecture="VR",
        ),
        ModelPreset(
            name="De-Echo/DeReverb",
            model_filename="deverb_bs_roformer_8_384dim_10depth.ckpt",
            description=(
                "Removes room reverb and echo using BS-Roformer (8-band, 384-dim, 10-layer).\n"
                "Best for: voice recorded in a room, interviews, speech with room sound,\n"
                "conference calls, podcast recordings in untreated spaces.\n"
                "Dry stem = close/direct signal with reverb removed — this is what you want.\n"
                "Reverb stem = extracted room sound (keep for creative use or discard).\n"
                "Tip: Overlap 12–16 gives cleanest room removal. Works on both speech and music."
            ),
            category="Audio Cleanup",
            stems=["dry", "reverb"],
            architecture="Roformer",
        ),
        ModelPreset(
            name="UVR DeEcho-DeReverb",
            model_filename="UVR-DeEcho-DeReverb.pth",
            description=(
                "VR architecture model for echo and reverb removal.\n"
                "Best for: lighter reverb reduction, natural-sounding cleanup.\n"
                "Dry stem = cleaned signal with reverb reduced. Reverb = extracted room.\n"
                "Less aggressive than the Roformer variant — preserves more room character.\n"
                "Tip: Try Window 320 for best clarity. Good when BS-Roformer is over-aggressive."
            ),
            category="Audio Cleanup",
            stems=["dry", "reverb"],
            architecture="VR",
        ),
    ],
}


def get_all_presets() -> list[ModelPreset]:
    """Return a flat list of all presets."""
    result = []
    for presets in MODEL_PRESETS.values():
        result.extend(presets)
    return result


def get_preset_by_name(name: str) -> Optional[ModelPreset]:
    """Find a preset by its display name."""
    for preset in get_all_presets():
        if preset.name == name:
            return preset
    return None


class SeparationWorker(QThread):
    """Worker thread for audio separation."""
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(list)  # output file paths
    error = pyqtSignal(str)

    def __init__(self, input_path: str, output_dir: str, preset: ModelPreset,
                 output_format: str = "wav",
                 extra_params: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.preset = preset
        self.output_format = output_format
        self.extra_params = extra_params or {}

    def run(self):
        try:
            self.progress.emit(5, f"Loading model: {self.preset.name}...")

            from audio_separator.separator import Separator

            separator = Separator(
                output_dir=self.output_dir,
                output_format=self.output_format,
                model_file_dir=_separator_model_dir(),
                **self.extra_params,
            )

            self.progress.emit(15, "Model loaded. Downloading if needed...")
            separator.load_model(model_filename=self.preset.model_filename)

            self.progress.emit(30, "Separating audio... (this may take a while)")
            output_files = separator.separate(self.input_path)

            # Ensure all paths are absolute (separator may return just filenames)
            resolved = []
            for f in output_files:
                p = Path(f)
                if not p.is_absolute():
                    p = Path(self.output_dir) / p
                resolved.append(str(p.resolve()))

            self.progress.emit(100, "Separation complete!")
            self.finished.emit(resolved)

        except Exception as e:
            self.error.emit(str(e))


class BatchSeparationWorker(QThread):
    """Worker thread for batch audio separation."""
    progress = pyqtSignal(int, str)
    file_finished = pyqtSignal(str, list)  # input_path, output_files
    all_finished = pyqtSignal()
    error = pyqtSignal(str, str)  # input_path, error message

    def __init__(self, input_paths: list[str], output_dir: str,
                 preset: ModelPreset, output_format: str = "wav", parent=None):
        super().__init__(parent)
        self.input_paths = input_paths
        self.output_dir = output_dir
        self.preset = preset
        self.output_format = output_format

    def run(self):
        try:
            from audio_separator.separator import Separator

            self.progress.emit(0, f"Loading model: {self.preset.name}...")

            separator = Separator(
                output_dir=self.output_dir,
                output_format=self.output_format,
                model_file_dir=_separator_model_dir(),
            )
            separator.load_model(model_filename=self.preset.model_filename)

            total = len(self.input_paths)
            for i, path in enumerate(self.input_paths):
                pct = int((i / total) * 100)
                self.progress.emit(pct, f"Processing {i+1}/{total}: {Path(path).name}")
                try:
                    output_files = separator.separate(path)
                    resolved = []
                    for f in output_files:
                        p = Path(f)
                        if not p.is_absolute():
                            p = Path(self.output_dir) / p
                        resolved.append(str(p.resolve()))
                    self.file_finished.emit(path, resolved)
                except Exception as e:
                    self.error.emit(path, str(e))

            self.progress.emit(100, "Batch processing complete!")
            self.all_finished.emit()

        except Exception as e:
            self.error.emit("", str(e))
