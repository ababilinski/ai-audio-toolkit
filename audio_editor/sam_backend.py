"""SAM-Audio (Facebook) integration for text-prompted audio separation."""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .runtime import ensure_ffmpeg_environment, get_ffmpeg_bin_dir

log = logging.getLogger(__name__)

# Chunking defaults (in seconds)
DEFAULT_CHUNK_DURATION = 30  # process 30s at a time
DEFAULT_OVERLAP_DURATION = 10  # 10s overlap for crossfade (Hann window)
DEFAULT_CROSSFADE_TYPE = "hann"  # "hann" or "linear"
DEFAULT_OVERLAP_ENABLED = True


def _hf_cache_dir() -> str:
    """Resolve the managed Hugging Face cache directory."""
    from .settings import get_app_settings

    path = Path(get_app_settings().hf_cache_dir())
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(path)
    return str(path)


def _ensure_ffmpeg_dlls():
    """Add FFmpeg shared DLL directory so torchcodec can load."""
    ffmpeg_bin_dir = ensure_ffmpeg_environment() or get_ffmpeg_bin_dir()
    if sys.platform == "win32" and ffmpeg_bin_dir is not None:
        try:
            os.add_dll_directory(str(ffmpeg_bin_dir))
        except (FileNotFoundError, OSError):
            pass


# Deferred availability check — don't import sam_audio at module level
# because torch DLLs may not be ready yet.
_sam_checked = False
SAM_AVAILABLE = False
SAM_IMPORT_ERROR = ""


def _patch_sam_base_model():
    """Patch sam_audio BaseModel._from_pretrained for newer huggingface_hub compatibility."""
    try:
        from sam_audio.model.base import BaseModel
        import inspect
        sig = inspect.signature(BaseModel._from_pretrained)
        for param_name in ("proxies", "resume_download"):
            param = sig.parameters.get(param_name)
            if param and param.default is inspect.Parameter.empty:
                orig = BaseModel._from_pretrained

                @classmethod
                def _patched_from_pretrained(
                    cls, *, model_id, cache_dir=None, force_download=False,
                    proxies=None, resume_download=False, local_files_only=False,
                    token=None, **kwargs,
                ):
                    return orig.__func__(
                        cls, model_id=model_id, cache_dir=cache_dir,
                        force_download=force_download, proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only, token=token, **kwargs,
                    )

                BaseModel._from_pretrained = _patched_from_pretrained
                break
    except Exception:
        pass


def _patch_judge_repo():
    """Redirect facebook/sam-audio-judge to the mrfakename mirror."""
    try:
        from sam_audio.model.config import JudgeRankerConfig
        _orig_judge_init = JudgeRankerConfig.__init__

        def _patched_judge_init(self, checkpoint_or_model_id="mrfakename/sam-audio-judge"):
            if "facebook/" in checkpoint_or_model_id:
                checkpoint_or_model_id = "mrfakename/sam-audio-judge"
            _orig_judge_init(self, checkpoint_or_model_id)

        JudgeRankerConfig.__init__ = _patched_judge_init
    except Exception:
        pass


def check_sam_available() -> bool:
    """Check if sam-audio can be imported. Safe to call multiple times."""
    global _sam_checked, SAM_AVAILABLE, SAM_IMPORT_ERROR
    if _sam_checked:
        return SAM_AVAILABLE
    _sam_checked = True
    try:
        _ensure_ffmpeg_dlls()
        import torch  # must load before sam_audio on Windows
        from sam_audio import SAMAudio, SAMAudioProcessor  # noqa: F401
        _patch_sam_base_model()
        SAM_AVAILABLE = True
    except Exception as e:
        SAM_IMPORT_ERROR = str(e)
    return SAM_AVAILABLE


def get_vram_budget() -> dict:
    """Return available VRAM info and recommended SAM settings."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"total_gb": 0, "free_gb": 0, "max_rerank": 1, "max_chunk_s": 30}

        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free = total - reserved

        # SAM base model ~4GB, each reranking candidate adds ~1.5GB for 30s chunk
        base_model_gb = 4.0
        per_candidate_gb = 1.5
        available = free - base_model_gb

        max_rerank = max(1, int(available / per_candidate_gb))
        # Reduce chunk duration when VRAM is tight
        if available > 3:
            max_chunk_s = 30
        elif available > 1.5:
            max_chunk_s = 15
        else:
            max_chunk_s = 10

        return {
            "total_gb": round(total, 1),
            "free_gb": round(free, 1),
            "max_rerank": min(max_rerank, 8),
            "max_chunk_s": max_chunk_s,
        }
    except Exception:
        return {"total_gb": 0, "free_gb": 0, "max_rerank": 1, "max_chunk_s": 30}


class SAMModelManager:
    """Singleton manager that loads the SAM-Audio model once and reuses it."""
    _instance = None
    _model = None
    _processor = None
    _loaded = False
    _model_name = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_name: str = "mrfakename/sam-audio-large"):
        if self._loaded and self._model_name == model_name:
            return
        _ensure_ffmpeg_dlls()
        import torch
        from sam_audio import SAMAudio, SAMAudioProcessor

        _patch_sam_base_model()
        _patch_judge_repo()

        try:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                cache_dir = _hf_cache_dir()
                self._processor = SAMAudioProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                )
                self._model = SAMAudio.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                )
        except Exception as e:
            err = str(e)
            if "gated repo" in err.lower() or "awaiting a review" in err.lower():
                raise RuntimeError(
                    f"Access to '{model_name}' is pending approval.\n\n"
                    "Please visit the model page on huggingface.co,\n"
                    "request access, and wait for the authors to approve.\n\n"
                    "Make sure you are logged in: huggingface-cli login"
                ) from e
            raise
        self._model = self._model.eval()
        if torch.cuda.is_available():
            # Clear any leftover VRAM before loading to GPU
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            try:
                self._model = self._model.cuda()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    # Report available VRAM for debugging
                    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    free = (total - torch.cuda.memory_reserved(0) / (1024**3))
                    raise RuntimeError(
                        f"Not enough GPU memory to load SAM-Audio model.\n"
                        f"Available: {free:.1f}GB / {total:.1f}GB total.\n\n"
                        f"Try:\n"
                        f"- Close other GPU applications\n"
                        f"- Use a smaller model (SAM-Audio Small)\n"
                        f"- Reduce re-ranking candidates"
                    ) from e
                raise
        self._model_name = model_name
        self._loaded = True

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def is_loaded(self):
        return self._loaded

    def unload(self):
        """Unload model from GPU to free VRAM."""
        import torch
        self._model = None
        self._processor = None
        self._loaded = False
        self._model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()


def find_checkpoint(input_path: str, output_dir: str, description: str,
                     model_name: str) -> str | None:
    """Check if a resumable checkpoint exists for this separation job."""
    input_stem = Path(input_path).stem
    ckpt_dir = Path(output_dir) / f".sam_chunks_{input_stem}"
    progress_file = ckpt_dir / "progress.json"
    if not progress_file.exists():
        return None
    try:
        with open(progress_file) as f:
            meta = json.load(f)
        if (meta.get("input_path") == input_path
                and meta.get("description") == description
                and meta.get("model_name") == model_name):
            completed = len(meta.get("completed", []))
            total = meta.get("total_chunks", 0)
            if 0 < completed < total:
                return str(ckpt_dir)
    except Exception:
        pass
    return None


class SAMSeparationWorker(QThread):
    """Worker thread for SAM-Audio text-prompted separation with chunked processing."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)  # output file paths
    error = pyqtSignal(str)

    def __init__(self, input_path: str, output_dir: str, description: str,
                 predict_spans: bool = False, reranking_candidates: int = 1,
                 model_name: str = "mrfakename/sam-audio-large",
                 chunk_duration: int = DEFAULT_CHUNK_DURATION,
                 overlap_duration: int = DEFAULT_OVERLAP_DURATION,
                 overlap_enabled: bool = DEFAULT_OVERLAP_ENABLED,
                 crossfade_type: str = DEFAULT_CROSSFADE_TYPE,
                 resume_checkpoint: str | None = None,
                 masked_video_path: str | None = None,
                 parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.description = description
        self.predict_spans = predict_spans
        self.reranking_candidates = reranking_candidates
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration if overlap_enabled else 0
        self.overlap_enabled = overlap_enabled
        self.crossfade_type = crossfade_type
        self.resume_checkpoint = resume_checkpoint
        self.masked_video_path = masked_video_path

    def run(self):
        try:
            _ensure_ffmpeg_dlls()
            import torch
            import torchaudio

            self.progress.emit(5, f"Loading SAM-Audio model ({self.model_name})...")
            manager = SAMModelManager.get()
            manager.load_model(self.model_name)

            sr = manager.processor.audio_sampling_rate  # 48000
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load full audio at SAM's sample rate
            self.progress.emit(10, "Loading audio...")
            wav, orig_sr = torchaudio.load(self.input_path)
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)
            # Mix to mono (SAM expects mono)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)  # (samples,)

            total_samples = wav.shape[0]
            duration_s = total_samples / sr
            log.info(f"Audio: {duration_s:.1f}s, {total_samples} samples @ {sr}Hz")

            # Decide whether to chunk
            min_chunk_threshold = self.chunk_duration + 10
            if duration_s <= min_chunk_threshold:
                # Short audio — process in one shot
                target, residual = self._separate_single(
                    manager, wav, device, sr, progress_start=15, progress_end=80
                )
            else:
                # Long audio — process in overlapping chunks
                target, residual = self._separate_chunked(
                    manager, wav, device, sr, duration_s
                )

            self.progress.emit(85, "Saving results...")

            # Build output file names
            safe_desc = self.description.replace(" ", "_").replace("/", "_")[:50]
            input_stem = Path(self.input_path).stem
            os.makedirs(self.output_dir, exist_ok=True)

            target_path = str(Path(self.output_dir) / f"{input_stem}_(SAM_{safe_desc})_target.wav")
            residual_path = str(Path(self.output_dir) / f"{input_stem}_(SAM_{safe_desc})_residual.wav")

            log.info(f"SAM target shape: {target.shape}, "
                     f"max: {target.abs().max():.4f}, "
                     f"mean: {target.abs().mean():.6f}")
            log.info(f"SAM residual shape: {residual.shape}, "
                     f"max: {residual.abs().max():.4f}, "
                     f"mean: {residual.abs().mean():.6f}")

            # Save as mono WAV — shape (1, samples)
            torchaudio.save(target_path, target.unsqueeze(0).cpu(), sr)
            torchaudio.save(residual_path, residual.unsqueeze(0).cpu(), sr)

            log.info(f"Saved: {target_path}")
            log.info(f"Saved: {residual_path}")

            # Free GPU memory
            del target, residual, wav
            manager.unload()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.progress.emit(100, "SAM separation complete!")
            self.finished.emit([target_path, residual_path])

        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            self.error.emit(str(e))

    def _separate_single(self, manager, wav, device, sr,
                         progress_start=15, progress_end=80):
        """Process audio in a single pass."""
        import torch

        self.progress.emit(progress_start, f"Separating: '{self.description}'...")

        proc_kwargs = dict(
            audios=[wav.unsqueeze(0)],
            descriptions=[self.description],
        )
        if self.masked_video_path:
            proc_kwargs["masked_videos"] = [self.masked_video_path]

        batch = manager.processor(**proc_kwargs).to(device)

        with torch.inference_mode():
            result = manager.model.separate(
                batch,
                predict_spans=self.predict_spans,
                reranking_candidates=self.reranking_candidates,
            )

        self.progress.emit(progress_end, "Processing results...")

        target = result.target[0].cpu().flatten()
        residual = result.residual[0].cpu().flatten()

        del result, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return target, residual

    def _get_checkpoint_dir(self) -> Path:
        """Return checkpoint directory path for this separation job."""
        input_stem = Path(self.input_path).stem
        return Path(self.output_dir) / f".sam_chunks_{input_stem}"

    def _save_chunk_checkpoint(self, ci, chunk_target, chunk_residual, chunks, metadata):
        """Save a completed chunk to disk for resume capability."""
        import torch
        ckpt_dir = self._get_checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(chunk_target, ckpt_dir / f"chunk_{ci}_target.pt")
        torch.save(chunk_residual, ckpt_dir / f"chunk_{ci}_residual.pt")

        metadata["completed"].append(ci)
        with open(ckpt_dir / "progress.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_checkpoint_metadata(self) -> dict | None:
        """Load checkpoint metadata if it exists and matches current params."""
        ckpt_dir = self._get_checkpoint_dir()
        progress_file = ckpt_dir / "progress.json"
        if not progress_file.exists():
            return None
        try:
            with open(progress_file) as f:
                meta = json.load(f)
            # Verify params match
            if (meta.get("input_path") == self.input_path
                    and meta.get("description") == self.description
                    and meta.get("model_name") == self.model_name):
                return meta
        except Exception:
            pass
        return None

    def _cleanup_checkpoint(self):
        """Remove checkpoint directory after successful completion."""
        ckpt_dir = self._get_checkpoint_dir()
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def _separate_chunked(self, manager, wav, device, sr, duration_s):
        """Process long audio in overlapping chunks with crossfade stitching."""
        import math
        import torch

        chunk_samples = self.chunk_duration * sr
        overlap_samples = self.overlap_duration * sr
        step_samples = chunk_samples - overlap_samples if overlap_samples > 0 else chunk_samples
        total_samples = wav.shape[0]

        # Calculate chunk positions
        chunks = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunks.append((start, end))
            if end >= total_samples:
                break
            start += step_samples

        n_chunks = len(chunks)
        log.info(f"Chunked processing: {n_chunks} chunks "
                 f"({self.chunk_duration}s each, {self.overlap_duration}s overlap, "
                 f"crossfade: {self.crossfade_type})")

        # Checkpoint metadata
        metadata = {
            "input_path": self.input_path,
            "description": self.description,
            "model_name": self.model_name,
            "total_chunks": n_chunks,
            "chunk_duration": self.chunk_duration,
            "overlap_duration": self.overlap_duration,
            "completed": [],
        }

        # Check for existing checkpoint to resume from
        completed_chunks = {}
        if self.resume_checkpoint:
            existing_meta = self._load_checkpoint_metadata()
            if existing_meta and existing_meta.get("total_chunks") == n_chunks:
                metadata["completed"] = existing_meta["completed"]
                ckpt_dir = self._get_checkpoint_dir()
                for ci in existing_meta["completed"]:
                    target_file = ckpt_dir / f"chunk_{ci}_target.pt"
                    residual_file = ckpt_dir / f"chunk_{ci}_residual.pt"
                    if target_file.exists() and residual_file.exists():
                        completed_chunks[ci] = (
                            torch.load(target_file, weights_only=True),
                            torch.load(residual_file, weights_only=True),
                        )
                log.info(f"Resuming: {len(completed_chunks)}/{n_chunks} chunks loaded")

        # Process each chunk
        target_full = torch.zeros(total_samples)
        residual_full = torch.zeros(total_samples)
        weight_full = torch.zeros(total_samples)

        for ci, (start, end) in enumerate(chunks):
            chunk_len = end - start
            pct = int(15 + (ci / n_chunks) * 65)

            # Use cached chunk if available
            if ci in completed_chunks:
                chunk_target, chunk_residual = completed_chunks[ci]
                self.progress.emit(pct,
                    f"Chunk {ci + 1}/{n_chunks}: loaded from checkpoint")
            else:
                self.progress.emit(pct,
                    f"Chunk {ci + 1}/{n_chunks}: "
                    f"{start / sr:.0f}s-{end / sr:.0f}s — '{self.description}'")

                chunk_wav = wav[start:end]

                proc_kwargs = dict(
                    audios=[chunk_wav.unsqueeze(0)],
                    descriptions=[self.description],
                )
                # For visual prompting with chunks, pass the full masked video
                # (the model's vision encoder extracts features per-chunk internally)
                if self.masked_video_path:
                    proc_kwargs["masked_videos"] = [self.masked_video_path]

                batch = manager.processor(**proc_kwargs).to(device)

                with torch.inference_mode():
                    result = manager.model.separate(
                        batch,
                        predict_spans=self.predict_spans,
                        reranking_candidates=self.reranking_candidates,
                    )

                chunk_target = result.target[0].cpu().flatten()
                chunk_residual = result.residual[0].cpu().flatten()

                del result, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Save checkpoint
                self._save_chunk_checkpoint(ci, chunk_target, chunk_residual,
                                            chunks, metadata)

            # Trim to actual chunk length (model may pad)
            chunk_target = chunk_target[:chunk_len]
            chunk_residual = chunk_residual[:chunk_len]

            # Build crossfade window
            window = torch.ones(chunk_len)
            if self.overlap_enabled and overlap_samples > 0:
                if ci > 0:
                    fade_len = min(overlap_samples, chunk_len)
                    if self.crossfade_type == "hann":
                        window[:fade_len] = 0.5 * (1 - torch.cos(
                            torch.linspace(0, math.pi, fade_len)))
                    else:
                        window[:fade_len] = torch.linspace(0, 1, fade_len)
                if ci < n_chunks - 1 and end < total_samples:
                    fade_len = min(overlap_samples, chunk_len)
                    if self.crossfade_type == "hann":
                        window[-fade_len:] = 0.5 * (1 + torch.cos(
                            torch.linspace(0, math.pi, fade_len)))
                    else:
                        window[-fade_len:] = torch.linspace(1, 0, fade_len)

            target_full[start:start + chunk_len] += chunk_target * window
            residual_full[start:start + chunk_len] += chunk_residual * window
            weight_full[start:start + chunk_len] += window

            del chunk_target, chunk_residual

        # Normalize by overlap weights (ensures proper blending)
        weight_full = weight_full.clamp(min=1e-8)
        target_full /= weight_full
        residual_full /= weight_full

        # Clean up checkpoint after successful completion
        self._cleanup_checkpoint()

        return target_full, residual_full
