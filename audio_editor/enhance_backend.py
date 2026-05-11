"""Audio enhancement backends (Resemble-Enhance)."""
import contextlib
import importlib.util
import logging
import math
import os
import sys
import time
import types
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

_BACKEND_UI_MODULES = {
    "resemble": "resemble_enhance",
    "deepfilter": "df.enhance",
    "clearvoice": "clearvoice",
    "audio_separator": "audio_separator.separator",
    "voicefixer": "voicefixer",
    "metricgan": "speechbrain.inference.enhancement",
}

# ── Availability ──

_checked = False
ENHANCE_AVAILABLE = False
ENHANCE_IMPORT_ERROR = ""
_download_patched = False


def _hf_cache_dir() -> str:
    """Resolve the managed Hugging Face cache directory."""
    from .settings import get_app_settings

    path = Path(get_app_settings().hf_cache_dir())
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(path)
    return str(path)


def _module_spec_exists(module_name: str) -> bool:
    """Return True when a module can be found without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def get_backend_probe_state(kind: str) -> dict[str, object]:
    """Return a lightweight backend state for UI notices without importing it."""
    states = {
        "resemble": (_checked, ENHANCE_AVAILABLE, ENHANCE_IMPORT_ERROR),
        "deepfilter": (_deepfilter_checked, DEEPFILTER_AVAILABLE, DEEPFILTER_IMPORT_ERROR),
        "clearvoice": (_clearvoice_checked, CLEARVOICE_AVAILABLE, CLEARVOICE_IMPORT_ERROR),
        "audio_separator": (
            _audio_separator_checked,
            AUDIO_SEPARATOR_AVAILABLE,
            AUDIO_SEPARATOR_IMPORT_ERROR,
        ),
        "voicefixer": (_voicefixer_checked, VOICEFIXER_AVAILABLE, VOICEFIXER_IMPORT_ERROR),
        "metricgan": (_metricgan_checked, METRICGAN_AVAILABLE, METRICGAN_IMPORT_ERROR),
    }
    if kind not in states:
        raise KeyError(f"Unknown backend probe kind: {kind}")

    checked, available, error = states[kind]
    return {
        "checked": bool(checked),
        "available": bool(available),
        "error": str(error or ""),
        "installed": _module_spec_exists(_BACKEND_UI_MODULES[kind]),
    }


def _deepfilter_cache_dir() -> str:
    """Resolve the managed DeepFilterNet cache root."""
    from .settings import get_app_settings

    path = Path(get_app_settings().deepfilter_model_dir())
    path.mkdir(parents=True, exist_ok=True)
    os.environ["AI_AUDIO_TOOLKIT_DEEPFILTER_MODEL_DIR"] = str(path)
    os.environ["AUDIO_EDITOR_DEEPFILTER_MODEL_DIR"] = str(path)
    return str(path)


def _separator_model_dir() -> str:
    """Resolve the managed audio-separator model directory."""
    from .settings import get_app_settings

    path = Path(get_app_settings().separator_model_dir())
    path.mkdir(parents=True, exist_ok=True)
    os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = str(path)
    return str(path)


def _clearvoice_cache_dir() -> str:
    """Resolve the managed ClearVoice checkpoint root."""
    from .settings import default_data_root

    path = default_data_root() / "models" / "ClearVoice"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _install_deepspeed_stub():
    """Install a minimal deepspeed stub so resemble-enhance can import.

    Deepspeed is only used for training — inference doesn't need it.
    The stub satisfies import-time references in train.py and utils/.
    """
    if "deepspeed" in sys.modules:
        return

    ds = types.ModuleType("deepspeed")
    ds.__path__ = []

    ds_acc = types.ModuleType("deepspeed.accelerator")
    ds_acc.get_accelerator = lambda: type(
        "Acc", (), {"communication_backend_name": lambda self: "nccl"}
    )()

    ds_runtime = types.ModuleType("deepspeed.runtime")
    ds_runtime.__path__ = []
    ds_engine = types.ModuleType("deepspeed.runtime.engine")
    ds_engine.DeepSpeedEngine = type("DeepSpeedEngine", (), {})
    ds_utils = types.ModuleType("deepspeed.runtime.utils")
    ds_utils.clip_grad_norm_ = lambda *a, **kw: None

    ds.DeepSpeedConfig = type(
        "DeepSpeedConfig", (), {"__init__": lambda self, *a, **kw: None}
    )
    ds.init_distributed = lambda *a, **kw: None
    ds.accelerator = ds_acc
    ds.runtime = ds_runtime
    ds.runtime.engine = ds_engine
    ds.runtime.utils = ds_utils

    for name, mod in [
        ("deepspeed", ds),
        ("deepspeed.accelerator", ds_acc),
        ("deepspeed.runtime", ds_runtime),
        ("deepspeed.runtime.engine", ds_engine),
        ("deepspeed.runtime.utils", ds_utils),
    ]:
        sys.modules[name] = mod


def _install_speechbrain_stubs():
    """Silence unavailable optional SpeechBrain integration modules.

    SpeechBrain uses a ``LazyModule`` system: every entry in
    ``speechbrain.integrations`` (k2_fsa, huggingface word-embeddings, …)
    is wrapped in a deferred loader.  When any code *accesses an attribute*
    of one of these lazy wrappers, SpeechBrain attempts the actual import.
    If the underlying optional package (k2, fairseq, fasttext, …) is absent
    the loader raises ``ImportError``, crashing even though MetricGAN+ never
    uses those integrations.

    Strategy (two-layered so we catch every case):
      1. Stub the known optional *backend* packages that the integrations
         depend on — so their modules import cleanly.
      2. Patch ``LazyModule.__getattr__`` so that any remaining unavailable
         integration raises ``AttributeError`` (which callers expect for
         missing attributes) rather than ``ImportError`` (which propagates
         as a hard crash).  SpectralMaskEnhancement never touches these
         integrations, so silencing them is safe.
    """
    # ── Layer 1: stub optional backend packages AND known failing integrations ──
    #
    # Two categories of stubs:
    #
    # a) Backend packages (k2, fairseq, fasttext …) — the optional C-extension
    #    libraries that integration modules try to import at module level.
    #    Stubbing them prevents the "ModuleNotFoundError: No module named 'k2'"
    #    error during disk-import of the integration .py file.
    #    NOTE: these stubs are intentionally empty; the integration modules they
    #    back are ALSO stubbed below (category b), so their attribute accesses
    #    never reach the backend stub at runtime.
    #
    # b) Integration modules themselves — pre-registering them in sys.modules
    #    means Python's import machinery returns our empty stub immediately,
    #    without ever executing the module's disk code.  This is the bullet-proof
    #    fix for any integration that imports missing optional packages at the
    #    TOP LEVEL of its .py file (wordemb imports fasttext; k2_fsa imports k2).
    #    Crucially, these stubs are installed BEFORE `import speechbrain.*` runs
    #    anywhere, so the LazyModule never even attempts the real import.
    _opt_backends = [
        "k2",
        "k2.ragged",
        "fairseq",
        "fairseq.models",
        "fairseq.tasks",
        "fasttext",
    ]
    # Known SpeechBrain integration modules that require optional deps.
    # Pre-stubbing the full dotted name (including parent packages) guarantees
    # Python never tries to execute the real .py file.
    _integration_stubs = [
        "speechbrain.k2_integration",
        "speechbrain.integrations",
        "speechbrain.integrations.k2_fsa",
        "speechbrain.integrations.huggingface",
        "speechbrain.integrations.huggingface.wordemb",
    ]
    for name in _opt_backends + _integration_stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
            log.debug("Installed speechbrain optional stub: %s", name)

    # ── Layer 2: patch LazyModule to raise AttributeError on failed loads ──
    try:
        import speechbrain.utils.importutils as _iu
    except Exception:
        return  # speechbrain not yet importable — skip the patch

    # Find the lazy-module class by duck-typing (name varies across versions)
    for _attr in dir(_iu):
        _cls = getattr(_iu, _attr, None)
        if not (isinstance(_cls, type) and "lazy" in _attr.lower()):
            continue
        if getattr(_cls, "_ae_patched", False):
            break  # already patched

        _orig_getattr = _cls.__getattr__

        def _tolerant_getattr(self, name, _orig=_orig_getattr):
            try:
                return _orig(self, name)
            except (ImportError, ModuleNotFoundError) as exc:
                # Convert to AttributeError so optional callers can catch it;
                # SpectralMaskEnhancement doesn't call these integrations.
                raise AttributeError(
                    f"Optional SpeechBrain integration unavailable "
                    f"({self!r}): {exc}"
                ) from exc

        _cls.__getattr__ = _tolerant_getattr
        _cls._ae_patched = True
        log.debug("Patched SpeechBrain LazyModule: %s", _attr)
        break


def _patch_posix_path():
    """Fix PosixPath deserialization in YAML on Windows.

    The resemble-enhance hparams.yaml was saved on Linux and contains
    ``!!python/object/apply:pathlib.PosixPath`` entries.  PosixPath cannot
    be instantiated on Windows.

    OmegaConf creates a fresh loader class in get_yaml_loader() each call
    and hardcodes PosixPath, so we must monkey-patch that function to fix
    the returned loader.
    """
    if sys.platform != "win32":
        return

    import omegaconf._utils as oc_utils
    _original_get_yaml_loader = oc_utils.get_yaml_loader

    def _patched_get_yaml_loader():
        loader = _original_get_yaml_loader()
        loader.add_constructor(
            "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
            lambda l, node: Path(*l.construct_sequence(node)),
        )
        return loader

    oc_utils.get_yaml_loader = _patched_get_yaml_loader
    log.info("Patched OmegaConf PosixPath constructor for Windows")


def _patch_cfm_numpy2():
    """Fix numpy 2.x incompatibility in resemble-enhance CFM solver.

    In cfm.py line 74, ``float(scipy.optimize.fsolve(...))`` fails with
    numpy >= 2.0 because ``float()`` on a 1-element ndarray was removed.
    We monkey-patch the ``exponential_decay_mapping`` static method to
    use ``.item()`` instead.
    """
    try:
        import numpy as np
        if np.lib.NumpyVersion(np.__version__) < "2.0.0":
            return

        from resemble_enhance.enhancer.lcfm.cfm import Solver
        import scipy.optimize

        @staticmethod
        def _fixed_mapping(t, n=4):
            def h(t, a):
                return (a**t - 1) / (a - 1)
            a = scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0).item()
            return h(t, a=a)

        Solver.exponential_decay_mapping = _fixed_mapping
        log.info("Patched CFM solver for numpy 2.x compatibility")
    except Exception as e:
        log.debug("CFM numpy2 patch skipped: %s", e)


def _patch_download():
    """Replace git-clone-based download with huggingface_hub snapshot_download.

    The upstream download.py uses git clone + git lfs which fails on Windows
    without git-lfs installed. huggingface_hub handles this natively.
    """
    global _download_patched
    if _download_patched:
        return
    _download_patched = True

    _patch_posix_path()
    _patch_cfm_numpy2()

    try:
        from huggingface_hub import snapshot_download
        import resemble_enhance.enhancer.download as dl_mod

        def _hf_download():
            log.info("Downloading resemble-enhance model via huggingface_hub...")
            local_dir = snapshot_download(
                repo_id="ResembleAI/resemble-enhance",
                cache_dir=_hf_cache_dir(),
                allow_patterns=["enhancer_stage2/**"],
            )
            run_dir = Path(local_dir) / "enhancer_stage2"
            log.info("Model downloaded to %s", run_dir)
            return run_dir

        dl_mod.download = _hf_download
        log.info("Patched resemble-enhance download to use huggingface_hub")
    except ImportError:
        log.warning("huggingface_hub not installed, falling back to git clone download")


def check_enhance_available() -> bool:
    """Check if resemble-enhance is importable."""
    global _checked, ENHANCE_AVAILABLE, ENHANCE_IMPORT_ERROR
    if _checked:
        return ENHANCE_AVAILABLE
    _checked = True
    try:
        _install_deepspeed_stub()
        _patch_download()
        from resemble_enhance.enhancer.inference import denoise, enhance  # noqa: F401
        ENHANCE_AVAILABLE = True
    except Exception as e:
        ENHANCE_IMPORT_ERROR = str(e)
        log.warning("resemble-enhance not available: %s", e)
    return ENHANCE_AVAILABLE


# ── DeepFilterNet availability ──

_deepfilter_checked = False
DEEPFILTER_AVAILABLE = False
DEEPFILTER_IMPORT_ERROR = ""

# ── VoiceFixer availability ──

_voicefixer_checked = False
VOICEFIXER_AVAILABLE = False
VOICEFIXER_IMPORT_ERROR = ""

# ── MetricGAN+ availability ──

_metricgan_checked = False
METRICGAN_AVAILABLE = False
METRICGAN_IMPORT_ERROR = ""

# ── ClearVoice availability ──

_clearvoice_checked = False
CLEARVOICE_AVAILABLE = False
CLEARVOICE_IMPORT_ERROR = ""

# ── audio-separator availability (used by MDX-Net enhancement) ──

_audio_separator_checked = False
AUDIO_SEPARATOR_AVAILABLE = False
AUDIO_SEPARATOR_IMPORT_ERROR = ""
_DEEPFILTER_CHUNK_SECONDS = 12
_DEEPFILTER_OVERLAP_SECONDS = 1.0
_DEEPFILTER_CHUNK_THRESHOLD_SECONDS = 18
_DEEPFILTER_GPU_TORCH_THREADS = 2
_DEEPFILTER_CPU_TORCH_THREADS = 4
_STUDIO_SEPARATOR_CHUNK_SECONDS = 90
VOICEFIXER_SUPPORTED_MODES = (0, 1, 2)
CLEARVOICE_MODEL_SAMPLE_RATES = {
    "MossFormer2_SE_48K": 48_000,
    "FRCRN_SE_16K": 16_000,
}
MDXNET_ENHANCEMENT_PRESETS = {
    "Kim Vocal 2": {
        "label": "Kim Vocal 2 (Recommended)",
        "description": (
            "Fast ONNX MDX-Net vocal isolator that works well for speech-forward cleanup.\n"
            "Best when the speaker is buried under room noise or crowd bleed."
        ),
    },
    "Kuielab Vocals": {
        "label": "Kuielab Vocals",
        "description": (
            "Lighter MDX-Net vocal model with a slightly different speech extraction character.\n"
            "Useful when Kim Vocal 2 sounds too aggressive."
        ),
    },
    "UVR MDX-NET Inst HQ 4": {
        "label": "UVR MDX-NET Inst HQ 4",
        "description": (
            "Stronger separation-oriented MDX-Net model that can pull speech out of dense backgrounds,\n"
            "though it may sound thinner than the other two."
        ),
    },
}
_MDXNET_PRIMARY_STEM_TOKENS = ("vocals", "vocal", "voice", "speech", "dialog", "dialogue")
_MDXNET_BACKGROUND_STEM_TOKENS = (
    "instrumental",
    "other",
    "background",
    "bleed",
    "room",
    "noise",
    "accompaniment",
)
_MDXNET_SEPARATOR_PARAMS = {
    "hop_length": 1024,
    "segment_size": 256,
    "overlap": 0.25,
    "batch_size": 1,
}
_DEREVERB_SEPARATOR_PARAMS = {
    "segment_size": 256,
    "overlap": 8,
    "batch_size": 1,
    "pitch_shift": 0,
}
_LEVELING_EPSILON = 1e-6
_LEVELING_PEAK_CEILING = 0.98
_LEVELING_MAX_AUTO_GAIN_DB = 18.0
_AUTO_LEVEL_TARGET_PEAK_DB = -1.0
_AUTO_LEVEL_MIN_ACTIVE_RMS = 0.10
_AUTO_LEVEL_MAX_ACTIVE_RMS = 0.18
_AUTO_LEVEL_MAX_WINDOW_GAIN = 4.5


def _install_deepfilter_compat():
    """Patch DeepFilterNet for newer torchaudio releases."""
    import sys
    import types
    from collections import namedtuple

    import soundfile as sf
    import torch  # noqa: F401  # load torch before torchaudio on Windows
    import torchaudio as ta

    if "torchaudio.backend.common" not in sys.modules:
        backend_pkg = sys.modules.get("torchaudio.backend")
        if backend_pkg is None:
            backend_pkg = types.ModuleType("torchaudio.backend")
            backend_pkg.__path__ = []
            sys.modules["torchaudio.backend"] = backend_pkg

        common_mod = types.ModuleType("torchaudio.backend.common")
        audio_meta = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )
        common_mod.AudioMetaData = audio_meta
        sys.modules["torchaudio.backend.common"] = common_mod

    if not hasattr(ta, "info"):
        AudioMetaData = sys.modules["torchaudio.backend.common"].AudioMetaData

        def _info(path: str, format: str | None = None):
            try:
                info = sf.info(path, format=format)
            except TypeError:
                info = sf.info(path)
            subtype = info.subtype or ""
            bits = 0
            for token in subtype.split("_"):
                if token.isdigit():
                    bits = int(token)
                    break
            return AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=bits,
                encoding=subtype or "UNKNOWN",
            )

        ta.info = _info


def _install_appdirs_stub(cache_dir: str) -> None:
    """Provide a minimal appdirs implementation for DeepFilterNet on Windows."""
    module = sys.modules.get("appdirs")
    if module is None:
        module = types.ModuleType("appdirs")
        sys.modules["appdirs"] = module

    root = str(Path(cache_dir).parent)

    def _user_cache_dir(appname: str | None = None, *args, **kwargs) -> str:
        if not appname:
            return root
        if appname == "DeepFilterNet":
            return cache_dir
        return str(Path(root) / appname)

    module.user_cache_dir = _user_cache_dir


def _patch_deepfilter_cache_dir() -> None:
    """Redirect DeepFilterNet downloads to the app-managed cache root."""
    cache_dir = _deepfilter_cache_dir()
    _install_appdirs_stub(cache_dir)

    def _managed_cache_dir() -> str:
        return cache_dir

    try:
        import df.utils as df_utils

        df_utils.get_cache_dir = _managed_cache_dir
    except Exception:
        pass

    try:
        import df.enhance as df_enhance

        df_enhance.get_cache_dir = _managed_cache_dir
        maybe_download_globals = getattr(df_enhance.maybe_download_model, "__globals__", {})
        if "get_cache_dir" in maybe_download_globals:
            maybe_download_globals["get_cache_dir"] = _managed_cache_dir
    except Exception:
        pass


def _load_deepfilter_runtime(*, post_filter: bool):
    """Load DeepFilterNet and normalize return shapes across versions."""
    _install_deepfilter_compat()
    _patch_deepfilter_cache_dir()
    from df.enhance import init_df, load_audio, enhance

    result = init_df(post_filter=post_filter)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("DeepFilterNet init_df() returned an unexpected result.")
    model, df_state = result[0], result[1]
    return model, df_state, enhance, load_audio


@contextlib.contextmanager
def _bounded_torch_threads():
    """Temporarily reduce PyTorch CPU thread usage for smoother desktop responsiveness."""
    try:
        import torch
    except Exception:
        yield
        return

    try:
        previous = torch.get_num_threads()
    except Exception:
        yield
        return

    target = _DEEPFILTER_GPU_TORCH_THREADS if torch.cuda.is_available() else _DEEPFILTER_CPU_TORCH_THREADS
    target = max(1, min(previous, target))
    changed = target != previous
    if changed:
        try:
            torch.set_num_threads(target)
        except Exception:
            changed = False
    try:
        yield
    finally:
        if changed:
            try:
                torch.set_num_threads(previous)
            except Exception:
                pass


def _clear_torch_resources() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


@contextlib.contextmanager
def _temporary_working_directory(path: str | Path):
    """Temporarily switch the process working directory."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(previous)


def _prepare_audio_for_model(
    input_path: str,
    work_dir: str | Path,
    *,
    target_sample_rate: int,
    mono: bool = True,
    stem_suffix: str = "prepared",
) -> tuple[str, dict[str, int]]:
    """Decode audio, resample it, and write a float32 WAV for a model."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    work_root = Path(work_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        audio_np, original_sample_rate = sf.read(input_path, always_2d=True)
        audio_np = np.asarray(audio_np, dtype=np.float32)
        original_channels = int(audio_np.shape[1])
    except Exception:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(input_path)
        original_sample_rate = int(segment.frame_rate)
        original_channels = int(segment.channels)
        sample_width = max(1, int(segment.sample_width))
        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
        if original_channels > 1:
            samples = samples.reshape(-1, original_channels)
        else:
            samples = samples.reshape(-1, 1)
        if sample_width == 1:
            audio_np = samples / 128.0
        elif sample_width == 2:
            audio_np = samples / 32768.0
        else:
            audio_np = samples / 2147483648.0

    waveform = torch.from_numpy(audio_np.T.copy())
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_sample_rate, target_sample_rate)

    prepared_path = work_root / f"{Path(input_path).stem}_{stem_suffix}_{target_sample_rate}hz.wav"
    prepared_audio = waveform.transpose(0, 1).contiguous().cpu().numpy()
    if prepared_audio.shape[1] == 1:
        prepared_audio = prepared_audio[:, 0]
    sf.write(prepared_path, prepared_audio, target_sample_rate, subtype="FLOAT")

    return str(prepared_path), {
        "original_sample_rate": original_sample_rate,
        "original_channels": original_channels,
        "prepared_sample_rate": int(target_sample_rate),
        "prepared_channels": int(waveform.shape[0]),
    }


def _normalize_audio_output(audio) -> "np.ndarray":
    """Normalize an enhancement result to a 1-D float32 numpy array."""
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()

    normalized = np.asarray(audio, dtype=np.float32).squeeze()
    if normalized.ndim == 0:
        raise RuntimeError("Audio enhancement returned an empty scalar output.")
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim == 2:
        axis = 0 if normalized.shape[0] <= normalized.shape[1] else 1
        return normalized.mean(axis=axis, dtype=np.float32)
    raise RuntimeError(f"Unsupported enhanced audio shape: {normalized.shape}")


def _coerce_audio_batch_tensor(audio):
    """Normalize enhanced tensor-like outputs to a (1, T) tensor."""
    import torch

    if hasattr(audio, "detach"):
        tensor = audio.detach().cpu()
    else:
        tensor = torch.as_tensor(audio)

    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 0:
        raise RuntimeError("Audio enhancement returned an empty scalar tensor.")
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise RuntimeError(f"Unsupported enhanced tensor shape: {tuple(tensor.shape)}")
    return tensor


def _write_mono_float_wav(
    output_path: str | Path,
    audio,
    sample_rate: int,
    *,
    target_sample_rate: int | None = None,
) -> int:
    """Write a mono float WAV, resampling first when requested."""
    import soundfile as sf
    import torch
    import torchaudio

    final_audio = _normalize_audio_output(audio)
    final_sample_rate = int(sample_rate)
    if target_sample_rate is not None and int(target_sample_rate) != final_sample_rate:
        tensor = torch.from_numpy(final_audio).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, final_sample_rate, int(target_sample_rate))
        final_audio = tensor.squeeze(0).cpu().numpy().astype("float32", copy=False)
        final_sample_rate = int(target_sample_rate)

    sf.write(str(output_path), final_audio, final_sample_rate, subtype="FLOAT")
    return final_sample_rate


def _clearvoice_output_sample_rate(model_name: str, original_sample_rate: int) -> int:
    """Choose the saved sample rate for a ClearVoice output."""
    native_rate = CLEARVOICE_MODEL_SAMPLE_RATES[model_name]
    if native_rate == 48_000:
        return native_rate
    return int(original_sample_rate) if int(original_sample_rate) > native_rate else native_rate


def _pip_install(package_spec: str, *, progress_callback=None, label: str = "") -> None:
    """Install a Python package via pip in the current interpreter.

    Runs ``sys.executable -m pip install <package_spec>`` so the package lands
    in the same environment the app is already running in.  Raises RuntimeError
    on failure so the calling worker can forward it to the UI via its error
    signal.

    In a PyInstaller frozen EXE build (``sys.frozen == True``) pip is not
    available.  The Windows bundle (build_bundle.py) ships a full CPython
    runtime where pip works fine, so auto-install is supported there.
    """
    import subprocess
    import sys

    if getattr(sys, "frozen", False):
        raise RuntimeError(
            f"{label or package_spec} is not bundled in this build.\n\n"
            "To use this engine, run the app from the Windows bundle "
            "(build_bundle.py) or from source, then press Enhance Audio "
            "again — it will install automatically."
        )

    display = label or package_spec
    if progress_callback:
        progress_callback(3, f"Installing {display} (first-time setup, please wait)...")
    log.info("Running: pip install %s", package_spec)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", package_spec],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-600:]
        raise RuntimeError(f"pip install {package_spec} failed:\n{tail}")
    log.info("pip install %s succeeded", package_spec)


# ── VoiceFixer chunking constants ──
_VOICEFIXER_CHUNK_SECONDS = 180        # 3 min per chunk when chunking is needed
_VOICEFIXER_OVERLAP_SECONDS = 2.0      # crossfade zone at each boundary
_VOICEFIXER_CHUNK_THRESHOLD_SECONDS = 240  # files shorter than 4 min run in one pass

# ── MetricGAN+ chunking constants ──
_METRICGAN_CHUNK_SECONDS = 30          # 30 s per chunk (at 16 kHz)
_METRICGAN_OVERLAP_SECONDS = 0.5       # crossfade zone at each boundary
_METRICGAN_CHUNK_THRESHOLD_SECONDS = 45   # files shorter than 45 s run in one pass


def _chunk_window(length: int, *, fade_in: bool, fade_out: bool, overlap_samples: int):
    import torch

    window = torch.ones(length, dtype=torch.float32)
    fade_len = min(overlap_samples, max(1, length // 2))
    if fade_len <= 0:
        return window
    if fade_in:
        window[:fade_len] = torch.linspace(0.0, 1.0, fade_len, dtype=window.dtype)
    if fade_out:
        fade = torch.linspace(1.0, 0.0, fade_len, dtype=window.dtype)
        if fade_len == length:
            window = torch.minimum(window, fade)
        else:
            window[-fade_len:] = torch.minimum(window[-fade_len:], fade)
    return window


def _enhance_audio_in_chunks(
    *,
    model,
    df_state,
    enhance_fn,
    audio,
    atten_lim_db: float,
    progress_callback=None,
    progress_start: int = 0,
    progress_end: int = 100,
):
    """Run DeepFilter enhancement in small overlapping chunks to avoid system stalls."""
    import torch

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    sample_rate = int(df_state.sr())
    total_samples = int(audio.shape[-1])
    total_seconds = total_samples / max(sample_rate, 1)
    if total_seconds <= _DEEPFILTER_CHUNK_THRESHOLD_SECONDS:
        if progress_callback:
            progress_callback(progress_start, "Processing enhancement...")
        with _bounded_torch_threads():
            enhanced = enhance_fn(model, df_state, audio, atten_lim_db=atten_lim_db)
        return enhanced.cpu()

    chunk_samples = max(sample_rate * _DEEPFILTER_CHUNK_SECONDS, sample_rate * 4)
    overlap_samples = int(sample_rate * _DEEPFILTER_OVERLAP_SECONDS)
    overlap_samples = min(overlap_samples, chunk_samples // 4)
    step_samples = max(1, chunk_samples - overlap_samples)

    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append((start, end))
        if end >= total_samples:
            break
        start += step_samples

    output = torch.zeros_like(audio, dtype=torch.float32)
    weights = torch.zeros(total_samples, dtype=torch.float32)
    num_chunks = len(chunks)

    with _bounded_torch_threads():
        for index, (start, end) in enumerate(chunks, start=1):
            if progress_callback:
                pct = progress_start + int(((index - 1) / max(1, num_chunks)) * max(1, progress_end - progress_start))
                progress_callback(
                    pct,
                    f"Processing noise chunk {index}/{num_chunks}..."
                )

            chunk_audio = audio[:, start:end]
            enhanced_chunk = enhance_fn(
                model,
                df_state,
                chunk_audio,
                atten_lim_db=atten_lim_db,
            ).cpu()
            enhanced_chunk = enhanced_chunk[..., : end - start].to(dtype=torch.float32)
            window = _chunk_window(
                end - start,
                fade_in=index > 1,
                fade_out=index < num_chunks,
                overlap_samples=overlap_samples,
            )
            output[:, start:end] += enhanced_chunk * window.unsqueeze(0)
            weights[start:end] += window

            del enhanced_chunk, chunk_audio
            _clear_torch_resources()
            time.sleep(0.01)

    weights = weights.clamp_min(1e-6)
    output = output / weights.unsqueeze(0)
    return output.cpu()


def _measure_rms(audio_data) -> float:
    import numpy as np

    audio_np = np.asarray(audio_data, dtype=np.float32)
    if audio_np.size == 0:
        return 0.0
    audio64 = audio_np.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(audio64), dtype=np.float64)))


def _to_mono(audio_data) -> "np.ndarray":
    import numpy as np

    audio_np = np.asarray(audio_data, dtype=np.float32)
    if audio_np.ndim == 2:
        return audio_np.mean(axis=1)
    return audio_np


def _window_rms_series(audio_data, sample_rate: int, *, window_ms: float = 50.0):
    import numpy as np

    mono = _to_mono(audio_data)
    if mono.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32), 1

    frame = max(1, int(sample_rate * (window_ms / 1000.0)))
    windows = []
    for start in range(0, len(mono), frame):
        chunk = mono[start:start + frame]
        if chunk.size == 0:
            continue
        windows.append(float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)), dtype=np.float64))))
    return np.asarray(windows, dtype=np.float32), frame


def _measure_active_rms(audio_data, sample_rate: int) -> float:
    import numpy as np

    window_rms, _ = _window_rms_series(audio_data, sample_rate)
    positive = window_rms[window_rms > _LEVELING_EPSILON]
    if positive.size == 0:
        return _measure_rms(audio_data)

    noise_floor = float(np.percentile(positive, 20))
    gate = max(float(np.percentile(positive, 35)), noise_floor * 1.8, 3e-4)
    active = positive[positive >= gate]
    if active.size == 0:
        return float(np.percentile(positive, 75))
    return float(np.mean(active))


def _apply_activity_auto_level(audio_data, sample_rate: int):
    import numpy as np
    from .effects import apply_compressor

    audio_np = np.asarray(audio_data, dtype=np.float32)
    if audio_np.size == 0:
        return audio_np, 0.0

    audio_2d = audio_np if audio_np.ndim == 2 else audio_np[:, np.newaxis]
    window_rms, frame = _window_rms_series(audio_2d, sample_rate)
    positive = window_rms[window_rms > _LEVELING_EPSILON]
    if positive.size < 4:
        return audio_np, 0.0

    noise_floor = float(np.percentile(positive, 20))
    gate = max(float(np.percentile(positive, 35)), noise_floor * 1.8, 3e-4)
    active_mask = window_rms >= gate
    if not np.any(active_mask):
        return audio_np, 0.0

    active = window_rms[active_mask]
    target_active_rms = float(np.percentile(active, 75) * 1.35)
    target_active_rms = float(np.clip(
        target_active_rms,
        _AUTO_LEVEL_MIN_ACTIVE_RMS,
        _AUTO_LEVEL_MAX_ACTIVE_RMS,
    ))

    desired = np.ones_like(window_rms, dtype=np.float32)
    desired[active_mask] = np.clip(
        target_active_rms / np.maximum(window_rms[active_mask], _LEVELING_EPSILON),
        0.70,
        _AUTO_LEVEL_MAX_WINDOW_GAIN,
    )

    nearby_mask = (~active_mask) & (window_rms >= gate * 0.45)
    if np.any(nearby_mask):
        nearby_gain = np.clip(
            target_active_rms / np.maximum(window_rms[nearby_mask], _LEVELING_EPSILON),
            0.85,
            2.2,
        )
        desired[nearby_mask] = 0.5 + 0.5 * nearby_gain

    smooth_windows = max(3, int(round(0.25 / max(frame / float(sample_rate), 1e-6))))
    kernel = np.ones(smooth_windows, dtype=np.float32) / smooth_windows
    smoothed = np.convolve(desired, kernel, mode="same")

    expanded = np.repeat(smoothed, frame)
    if expanded.size < audio_2d.shape[0]:
        expanded = np.pad(expanded, (0, audio_2d.shape[0] - expanded.size), mode="edge")
    else:
        expanded = expanded[:audio_2d.shape[0]]

    before_active_rms = _measure_active_rms(audio_2d, sample_rate)
    leveled = audio_2d * expanded[:, np.newaxis]

    compressed_windows, _ = _window_rms_series(leveled, sample_rate)
    active_windows = compressed_windows[compressed_windows > _LEVELING_EPSILON]
    if active_windows.size >= 4:
        threshold_rms = float(np.percentile(active_windows, 75) * 1.1)
        threshold_db = 20.0 * math.log10(max(threshold_rms, 1e-4))
        leveled = apply_compressor(
            leveled,
            sample_rate,
            threshold_db=threshold_db,
            ratio=3.5,
            attack_ms=5.0,
            release_ms=120.0,
        ).astype(np.float32)

    after_active_rms = _measure_active_rms(leveled, sample_rate)
    applied_gain_db = 0.0
    if before_active_rms > _LEVELING_EPSILON and after_active_rms > _LEVELING_EPSILON:
        applied_gain_db = 20.0 * math.log10(after_active_rms / before_active_rms)

    if audio_np.ndim == 1:
        return leveled[:, 0].astype(np.float32), applied_gain_db
    return leveled.astype(np.float32), applied_gain_db


def _apply_saved_audio_leveling(
    input_path: str,
    output_path: str,
    *,
    auto_level: bool,
    match_input_loudness: bool,
    output_gain_db: float,
    progress_callback=None,
    progress_pct: int = 92,
) -> None:
    """Optionally match loudness to the input and apply a clip-safe gain trim."""
    if not auto_level and not match_input_loudness and abs(output_gain_db) < 1e-6:
        return

    import numpy as np
    import soundfile as sf

    if progress_callback:
        progress_callback(progress_pct, "Balancing output level...")

    output_audio, output_sr = sf.read(output_path, always_2d=False)
    output_audio = np.asarray(output_audio, dtype=np.float32)
    if output_audio.size == 0:
        log.info("Skipping output leveling because the enhanced file is empty: %s", output_path)
        return

    applied_match_gain_db = 0.0
    if match_input_loudness:
        input_audio, input_sr = sf.read(input_path, always_2d=False)
        input_rms = _measure_active_rms(input_audio, input_sr)
        output_rms = _measure_active_rms(output_audio, output_sr)
        if input_rms <= _LEVELING_EPSILON:
            input_rms = _measure_rms(input_audio)
        if output_rms <= _LEVELING_EPSILON:
            output_rms = _measure_rms(output_audio)
        if input_rms > _LEVELING_EPSILON and output_rms > _LEVELING_EPSILON:
            applied_match_gain_db = 20.0 * math.log10(input_rms / output_rms)
            applied_match_gain_db = max(
                -_LEVELING_MAX_AUTO_GAIN_DB,
                min(_LEVELING_MAX_AUTO_GAIN_DB, applied_match_gain_db),
            )
            output_audio = output_audio * float(10.0 ** (applied_match_gain_db / 20.0))
        else:
            log.info(
                "Skipping automatic loudness matching for %s (input_rms=%s, output_rms=%s)",
                output_path,
                input_rms,
                output_rms,
            )

    applied_auto_level_db = 0.0
    if auto_level:
        output_audio, applied_auto_level_db = _apply_activity_auto_level(output_audio, output_sr)
        peak = float(np.max(np.abs(output_audio))) if output_audio.size else 0.0
        if peak > _LEVELING_EPSILON:
            target_peak = float(10.0 ** (_AUTO_LEVEL_TARGET_PEAK_DB / 20.0))
            peak_gain = target_peak / peak
            output_audio = output_audio * peak_gain
            applied_auto_level_db += 20.0 * math.log10(peak_gain)
        else:
            log.info("Skipping auto level peak trim for %s because peak is too small: %s", output_path, peak)

    manual_gain_db = float(output_gain_db)
    if abs(manual_gain_db) > 1e-6:
        output_audio = output_audio * float(10.0 ** (manual_gain_db / 20.0))

    peak = float(np.max(np.abs(output_audio))) if output_audio.size else 0.0
    limiter_gain_db = 0.0
    if peak > _LEVELING_PEAK_CEILING:
        limiter_gain = _LEVELING_PEAK_CEILING / peak
        output_audio = output_audio * limiter_gain
        limiter_gain_db = 20.0 * math.log10(limiter_gain)

    info = sf.info(output_path)
    sf.write(output_path, output_audio, output_sr, format=info.format, subtype=info.subtype)
    log.info(
        "Applied output leveling to %s (match=%+.2f dB, auto_level=%+.2f dB, manual=%+.2f dB, limiter=%+.2f dB)",
        output_path,
        applied_match_gain_db,
        applied_auto_level_db,
        manual_gain_db,
        limiter_gain_db,
    )


def check_deepfilter_available() -> bool:
    """Check if deepfilternet is importable."""
    global _deepfilter_checked, DEEPFILTER_AVAILABLE, DEEPFILTER_IMPORT_ERROR
    if _deepfilter_checked:
        return DEEPFILTER_AVAILABLE
    _deepfilter_checked = True
    try:
        _install_deepfilter_compat()
        _patch_deepfilter_cache_dir()
        from df.enhance import enhance, init_df  # noqa: F401
        DEEPFILTER_AVAILABLE = True
    except Exception as e:
        DEEPFILTER_IMPORT_ERROR = str(e)
        log.warning("deepfilternet not available: %s", e)
    return DEEPFILTER_AVAILABLE


def check_voicefixer_available() -> bool:
    """Check if voicefixer is importable."""
    global _voicefixer_checked, VOICEFIXER_AVAILABLE, VOICEFIXER_IMPORT_ERROR
    if _voicefixer_checked:
        return VOICEFIXER_AVAILABLE
    _voicefixer_checked = True
    try:
        from voicefixer import VoiceFixer  # noqa: F401
        VOICEFIXER_AVAILABLE = True
    except Exception as e:
        VOICEFIXER_IMPORT_ERROR = str(e)
        log.warning("voicefixer not available: %s", e)
    return VOICEFIXER_AVAILABLE


def check_metricgan_available() -> bool:
    """Check if speechbrain MetricGAN+ is importable."""
    global _metricgan_checked, METRICGAN_AVAILABLE, METRICGAN_IMPORT_ERROR
    if _metricgan_checked:
        return METRICGAN_AVAILABLE
    _metricgan_checked = True
    try:
        _install_speechbrain_stubs()
        from speechbrain.inference.enhancement import SpectralMaskEnhancement  # noqa: F401
        METRICGAN_AVAILABLE = True
    except Exception as e:
        METRICGAN_IMPORT_ERROR = str(e)
        log.warning("speechbrain/MetricGAN+ not available: %s", e)
    return METRICGAN_AVAILABLE


def check_clearvoice_available() -> bool:
    """Check if clearvoice is importable."""
    global _clearvoice_checked, CLEARVOICE_AVAILABLE, CLEARVOICE_IMPORT_ERROR
    if _clearvoice_checked:
        return CLEARVOICE_AVAILABLE
    _clearvoice_checked = True
    try:
        from clearvoice import ClearVoice  # noqa: F401
        CLEARVOICE_AVAILABLE = True
    except Exception as e:
        CLEARVOICE_IMPORT_ERROR = str(e)
        log.warning("clearvoice not available: %s", e)
    return CLEARVOICE_AVAILABLE


def check_audio_separator_available() -> bool:
    """Check if audio-separator is importable."""
    global _audio_separator_checked, AUDIO_SEPARATOR_AVAILABLE, AUDIO_SEPARATOR_IMPORT_ERROR
    if _audio_separator_checked:
        return AUDIO_SEPARATOR_AVAILABLE
    _audio_separator_checked = True
    try:
        from audio_separator.separator import Separator  # noqa: F401

        AUDIO_SEPARATOR_AVAILABLE = True
    except Exception as e:
        AUDIO_SEPARATOR_IMPORT_ERROR = str(e)
        log.warning("audio-separator not available: %s", e)
    return AUDIO_SEPARATOR_AVAILABLE


class DeepFilterWorker(QThread):
    """Worker thread for speech/vocal enhancement using DeepFilterNet."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, input_path: str, output_dir: str,
                 atten_lim_db: float = 70.0,
                 post_filter: bool = False,
                 auto_level: bool = True,
                 match_input_loudness: bool = True,
                 output_gain_db: float = 0.0,
                 parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.atten_lim_db = atten_lim_db
        self.post_filter = post_filter
        self.auto_level = auto_level
        self.match_input_loudness = match_input_loudness
        self.output_gain_db = output_gain_db

    def run(self):
        try:
            import numpy as np
            import soundfile as sf

            self.progress.emit(5, "Loading audio...")
            self.progress.emit(10, "Loading DeepFilterNet model (first run downloads ~15MB)...")

            model, df_state, enhance, load_audio = _load_deepfilter_runtime(
                post_filter=self.post_filter
            )

            self.progress.emit(25, "Resampling to 48kHz...")
            audio, _ = load_audio(self.input_path, sr=df_state.sr())

            self.progress.emit(40, f"Enhancing speech/vocals (attenuation limit: {self.atten_lim_db:.0f} dB)...")
            enhanced = _enhance_audio_in_chunks(
                model=model,
                df_state=df_state,
                enhance_fn=enhance,
                audio=audio,
                atten_lim_db=self.atten_lim_db,
                progress_callback=self.progress.emit,
                progress_start=40,
                progress_end=88,
            )

            os.makedirs(self.output_dir, exist_ok=True)
            input_stem = Path(self.input_path).stem
            output_path = str(Path(self.output_dir) / f"{input_stem}_deepfilter.wav")

            self.progress.emit(90, "Saving output...")
            # enhanced shape: (C, T) — transpose to (T, C) for soundfile
            audio_np = enhanced.cpu().numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T  # (T, C)
            elif audio_np.ndim == 1:
                pass  # (T,) is fine for mono
            sf.write(output_path, audio_np, df_state.sr())
            _apply_saved_audio_leveling(
                self.input_path,
                output_path,
                auto_level=self.auto_level,
                match_input_loudness=self.match_input_loudness,
                output_gain_db=self.output_gain_db,
                progress_callback=self.progress.emit,
                progress_pct=94,
            )
            log.info("Saved DeepFilterNet output: %s", output_path)

            self.progress.emit(100, "Enhancement complete!")
            self.finished.emit([output_path])

            # Cleanup
            del enhanced, audio, model
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            self.error.emit(str(e))


# ── Studio Sound Worker ──

# Best available De-Echo/DeReverb model from the current audio-separator preset list
_DEREVERB_MODEL = "deverb_bs_roformer_8_384dim_10depth.ckpt"


def _clamp_int(value, minimum: int, maximum: int, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(maximum, coerced))


def _clamp_float(value, minimum: float, maximum: float, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(maximum, coerced))


def _normalize_mdx_separator_params(
    *,
    segment_size: int = _MDXNET_SEPARATOR_PARAMS["segment_size"],
    overlap: float = _MDXNET_SEPARATOR_PARAMS["overlap"],
    batch_size: int = _MDXNET_SEPARATOR_PARAMS["batch_size"],
    enable_denoise: bool = True,
) -> dict[str, object]:
    return {
        "hop_length": _MDXNET_SEPARATOR_PARAMS["hop_length"],
        "segment_size": _clamp_int(segment_size, 32, 512, _MDXNET_SEPARATOR_PARAMS["segment_size"]),
        "overlap": _clamp_float(overlap, 0.01, 0.95, _MDXNET_SEPARATOR_PARAMS["overlap"]),
        "batch_size": _clamp_int(batch_size, 1, 16, _MDXNET_SEPARATOR_PARAMS["batch_size"]),
        "enable_denoise": bool(enable_denoise),
    }


def _normalize_dereverb_separator_params(
    *,
    segment_size: int = _DEREVERB_SEPARATOR_PARAMS["segment_size"],
    overlap: int = _DEREVERB_SEPARATOR_PARAMS["overlap"],
    batch_size: int = _DEREVERB_SEPARATOR_PARAMS["batch_size"],
    pitch_shift: int = _DEREVERB_SEPARATOR_PARAMS["pitch_shift"],
) -> dict[str, int]:
    return {
        "segment_size": _clamp_int(
            segment_size,
            32,
            512,
            _DEREVERB_SEPARATOR_PARAMS["segment_size"],
        ),
        "overlap": _clamp_int(
            overlap,
            1,
            32,
            _DEREVERB_SEPARATOR_PARAMS["overlap"],
        ),
        "batch_size": _clamp_int(
            batch_size,
            1,
            16,
            _DEREVERB_SEPARATOR_PARAMS["batch_size"],
        ),
        "pitch_shift": _clamp_int(
            pitch_shift,
            -12,
            12,
            _DEREVERB_SEPARATOR_PARAMS["pitch_shift"],
        ),
    }


def _run_dereverb_stage(
    input_path: str,
    output_dir: str | Path,
    final_output_path: str | Path,
    *,
    mdxc_params: dict | None = None,
    progress_callback=None,
    progress_start: int = 0,
    progress_end: int = 0,
) -> str:
    """Run the BS-Roformer dereverb model and keep the dry stem."""
    from pathlib import Path as _Path

    from audio_separator.separator import Separator

    output_root = _Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_mdxc_params = _normalize_dereverb_separator_params(
        **(mdxc_params or {})
    )

    if progress_callback:
        progress_callback(progress_start, "Removing room echo/reverb (De-Echo Roformer)...")

    separator = Separator(
        output_dir=str(output_root),
        model_file_dir=_separator_model_dir(),
        chunk_duration=_STUDIO_SEPARATOR_CHUNK_SECONDS,
        mdxc_params=resolved_mdxc_params,
    )
    separator.load_model(model_filename=_DEREVERB_MODEL)

    if progress_callback:
        midpoint = progress_start + max(1, (progress_end - progress_start) // 2)
        progress_callback(midpoint, "Processing reverb removal...")

    outputs = separator.separate(input_path)
    resolved_outputs = []
    for item in outputs:
        candidate = _Path(item)
        if not candidate.is_absolute():
            candidate = output_root / candidate
        resolved_outputs.append(str(candidate.resolve()))

    dry_stem = next(
        (
            item for item in resolved_outputs
            if any(token in item.lower() for token in ("no_reverb", "noreverb", "dry", "dereverb"))
        ),
        resolved_outputs[0] if resolved_outputs else None,
    )
    if dry_stem is None:
        raise RuntimeError("De-Echo model produced no output files.")

    final_path = _Path(final_output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _replace_file(dry_stem, final_path)

    if progress_callback:
        progress_callback(progress_end, "Reverb removal complete.")
    return str(final_path)


def _resolve_separator_output_paths(
    output_files: list[str],
    output_dir: str | Path,
) -> list[Path]:
    """Resolve audio-separator outputs to absolute filesystem paths."""
    output_root = Path(output_dir)
    resolved: list[Path] = []
    for item in output_files:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = output_root / candidate
        resolved.append(candidate.resolve())
    return resolved


def _pick_separator_output(
    output_files: list[Path],
    preferred_tokens: tuple[str, ...],
) -> Path | None:
    """Pick the best separator output by matching stem tokens in the filename."""
    normalized_tokens = tuple(
        token.lower().replace(" ", "").replace("-", "").replace("_", "")
        for token in preferred_tokens
        if token
    )
    for token in normalized_tokens:
        for candidate in output_files:
            normalized_name = (
                candidate.stem.lower().replace(" ", "").replace("-", "").replace("_", "")
            )
            if token in normalized_name:
                return candidate
    return output_files[0] if output_files else None


def _replace_file(source_path: str | Path, destination_path: str | Path) -> str:
    """Move a file to its destination, replacing any existing file."""
    import shutil

    source = Path(source_path)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            destination.unlink()
        except OSError:
            pass
    shutil.move(str(source), str(destination))
    return str(destination)


def _run_separator_stage(
    input_path: str,
    output_dir: str | Path,
    *,
    model_filename: str,
    separator_params: dict | None = None,
    progress_callback=None,
    progress_start: int = 0,
    progress_end: int = 0,
    loading_message: str = "Loading separator model...",
    processing_message: str = "Separating audio...",
    complete_message: str = "Separation complete.",
) -> list[Path]:
    """Run an audio-separator model and return all resolved output files."""
    from audio_separator.separator import Separator

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(progress_start, loading_message)

    separator_kwargs = dict(separator_params or {})
    separator = Separator(
        output_dir=str(output_root),
        output_format="wav",
        model_file_dir=_separator_model_dir(),
        chunk_duration=_STUDIO_SEPARATOR_CHUNK_SECONDS,
        **separator_kwargs,
    )
    separator.load_model(model_filename=model_filename)

    if progress_callback:
        midpoint = progress_start + max(1, (progress_end - progress_start) // 2)
        progress_callback(midpoint, processing_message)

    outputs = separator.separate(input_path)
    resolved_outputs = _resolve_separator_output_paths(outputs, output_root)
    if not resolved_outputs:
        raise RuntimeError("Separator model produced no output files.")

    if progress_callback:
        progress_callback(progress_end, complete_message)
    return resolved_outputs


def _run_separator_primary_stem_stage(
    input_path: str,
    output_dir: str | Path,
    final_output_path: str | Path,
    *,
    model_filename: str,
    separator_params: dict | None = None,
    preferred_tokens: tuple[str, ...] = (),
    progress_callback=None,
    progress_start: int = 0,
    progress_end: int = 0,
    loading_message: str = "Loading separator model...",
    processing_message: str = "Separating speech...",
    complete_message: str = "Speech isolation complete.",
) -> str:
    """Run an audio-separator model and keep the best matching primary stem."""
    resolved_outputs = _run_separator_stage(
        input_path,
        output_dir,
        model_filename=model_filename,
        separator_params=separator_params,
        progress_callback=progress_callback,
        progress_start=progress_start,
        progress_end=progress_end,
        loading_message=loading_message,
        processing_message=processing_message,
        complete_message=complete_message,
    )
    primary_output = _pick_separator_output(resolved_outputs, preferred_tokens)
    if primary_output is None:
        raise RuntimeError("Separator model produced no output files.")

    final_path = Path(final_output_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _replace_file(primary_output, final_path)

    for candidate in resolved_outputs:
        if candidate == primary_output:
            continue
        if candidate.is_file():
            try:
                candidate.unlink()
            except OSError:
                pass

    if progress_callback:
        progress_callback(progress_end, complete_message)
    return str(final_path)


class StudioSoundWorker(QThread):
    """Worker thread for 'Studio Sound' — chains DeepFilterNet (noise) + De-Echo Roformer (reverb).

    Each stage is optional. The output of each stage becomes the input of the next.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        remove_noise: bool = True,
        atten_lim_db: float = 70.0,
        post_filter: bool = False,
        remove_reverb: bool = True,
        dereverb_segment_size: int = _DEREVERB_SEPARATOR_PARAMS["segment_size"],
        dereverb_overlap: int = _DEREVERB_SEPARATOR_PARAMS["overlap"],
        dereverb_batch_size: int = _DEREVERB_SEPARATOR_PARAMS["batch_size"],
        dereverb_pitch_shift: int = _DEREVERB_SEPARATOR_PARAMS["pitch_shift"],
        auto_level: bool = True,
        match_input_loudness: bool = True,
        output_gain_db: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.remove_noise = remove_noise
        self.atten_lim_db = atten_lim_db
        self.post_filter = post_filter
        self.remove_reverb = remove_reverb
        self.dereverb_segment_size = dereverb_segment_size
        self.dereverb_overlap = dereverb_overlap
        self.dereverb_batch_size = dereverb_batch_size
        self.dereverb_pitch_shift = dereverb_pitch_shift
        self.auto_level = auto_level
        self.match_input_loudness = match_input_loudness
        self.output_gain_db = output_gain_db

    def run(self):
        try:
            import gc
            import soundfile as sf
            from pathlib import Path as _Path

            output_dir = _Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            input_stem = _Path(self.input_path).stem
            current = self.input_path
            total_steps = sum([self.remove_noise, self.remove_reverb])
            if total_steps == 0:
                self.error.emit("No processing steps selected. Enable at least one option.")
                return

            step = 0

            # ── Stage 1: Noise removal via DeepFilterNet ──
            if self.remove_noise:
                step += 1
                base_pct = int((step - 1) / total_steps * 90)
                end_pct = int(step / total_steps * 90)

                self.progress.emit(base_pct, "Removing background noise (DeepFilterNet)...")
                log.info("Studio Sound: DeepFilterNet noise removal")

                model, df_state, enhance, load_audio = _load_deepfilter_runtime(
                    post_filter=self.post_filter
                )
                audio, _ = load_audio(current, sr=df_state.sr())

                stage_mid_pct = base_pct + (end_pct - base_pct) // 2
                self.progress.emit(stage_mid_pct, "Processing noise in manageable chunks...")

                enhanced = _enhance_audio_in_chunks(
                    model=model,
                    df_state=df_state,
                    enhance_fn=enhance,
                    audio=audio,
                    atten_lim_db=self.atten_lim_db,
                    progress_callback=self.progress.emit,
                    progress_start=stage_mid_pct,
                    progress_end=max(stage_mid_pct + 1, end_pct - 2),
                )
                noise_out = str(output_dir / f"{input_stem}_denoised_tmp.wav")
                audio_np = enhanced.cpu().numpy()
                if audio_np.ndim == 2:
                    audio_np = audio_np.T
                sf.write(noise_out, audio_np, df_state.sr())
                current = noise_out

                del model, enhanced, audio
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                self.progress.emit(end_pct, "Noise removal complete.")

            # ── Stage 2: Reverb/echo removal via De-Echo Roformer ──
            if self.remove_reverb:
                step += 1
                base_pct = int((step - 1) / total_steps * 90)
                end_pct = int(step / total_steps * 90)

                log.info("Studio Sound: De-Echo Roformer reverb removal")
                current = _run_dereverb_stage(
                    current,
                    output_dir,
                    output_dir / f"{input_stem}_studio_sound.wav",
                    mdxc_params={
                        "segment_size": self.dereverb_segment_size,
                        "overlap": self.dereverb_overlap,
                        "batch_size": self.dereverb_batch_size,
                        "pitch_shift": self.dereverb_pitch_shift,
                    },
                    progress_callback=self.progress.emit,
                    progress_start=base_pct,
                    progress_end=end_pct,
                )

            # ── Clean up intermediate denoised temp file ──
            if self.remove_noise and self.remove_reverb:
                tmp = _Path(output_dir) / f"{input_stem}_denoised_tmp.wav"
                if tmp.is_file():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

            _apply_saved_audio_leveling(
                self.input_path,
                current,
                auto_level=self.auto_level,
                match_input_loudness=self.match_input_loudness,
                output_gain_db=self.output_gain_db,
                progress_callback=self.progress.emit,
                progress_pct=94,
            )

            self.progress.emit(100, "Studio Sound complete!")
            self.finished.emit([current])

        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            self.error.emit(str(e))


def _restore_voicefixer_in_chunks(
    vf,
    input_path: str,
    output_path: str,
    cuda: bool,
    mode: int,
    *,
    progress_callback=None,
    progress_start: int = 20,
    progress_end: int = 88,
) -> None:
    """Run VoiceFixer on a long file by splitting into overlapping chunks.

    Each chunk is written to a temp WAV, processed by ``vf.restore()``, then
    read back.  Adjacent chunks are crossfaded over ``_VOICEFIXER_OVERLAP_SECONDS``
    at the output sample-rate to avoid audible seams.
    """
    import tempfile

    import numpy as np
    import soundfile as sf

    info = sf.info(input_path)
    total_seconds = info.duration
    input_sr = info.samplerate

    # Short file — single pass
    if total_seconds <= _VOICEFIXER_CHUNK_THRESHOLD_SECONDS:
        if progress_callback:
            progress_callback(progress_start, "Restoring speech...")
        vf.restore(input=input_path, output=output_path, cuda=cuda, mode=mode)
        return

    audio, _ = sf.read(input_path, always_2d=True)   # (T, C)
    total_samples = len(audio)
    chunk_samples = int(input_sr * _VOICEFIXER_CHUNK_SECONDS)
    overlap_samples = int(input_sr * _VOICEFIXER_OVERLAP_SECONDS)
    step_samples = max(1, chunk_samples - overlap_samples)

    # Build chunk boundary list
    segs: list[tuple[int, int]] = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        segs.append((start, end))
        if end >= total_samples:
            break
        start += step_samples

    num_segs = len(segs)
    processed: list[np.ndarray] = []
    out_sr: int = input_sr

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for idx, (seg_start, seg_end) in enumerate(segs, start=1):
            if progress_callback:
                pct = progress_start + int(
                    ((idx - 1) / max(1, num_segs))
                    * max(1, progress_end - progress_start)
                )
                progress_callback(pct, f"Restoring chunk {idx}/{num_segs}...")

            chunk_in = str(tmp / f"in_{idx}.wav")
            chunk_out = str(tmp / f"out_{idx}.wav")
            sf.write(chunk_in, audio[seg_start:seg_end], input_sr)
            vf.restore(input=chunk_in, output=chunk_out, cuda=cuda, mode=mode)

            chunk_audio, chunk_sr = sf.read(chunk_out, always_2d=True)
            out_sr = int(chunk_sr)
            processed.append(chunk_audio)

    if not processed:
        raise RuntimeError("VoiceFixer produced no output for any chunk.")

    # Scale overlap to output sample rate
    out_overlap = max(1, int(overlap_samples * out_sr / max(input_sr, 1)))

    # Crossfade-merge all chunks
    result = processed[0]
    for i in range(1, len(processed)):
        prev = result
        curr = processed[i]
        # Trim the half-overlap tail from prev and head from curr, blend the shared zone
        xf_len = min(out_overlap, len(prev), len(curr))
        fade_out = np.linspace(1.0, 0.0, xf_len, dtype=np.float32)[:, np.newaxis]
        fade_in = np.linspace(0.0, 1.0, xf_len, dtype=np.float32)[:, np.newaxis]
        blended = prev[-xf_len:] * fade_out + curr[:xf_len] * fade_in
        result = np.concatenate([prev[:-xf_len], blended, curr[xf_len:]], axis=0)

    # Write merged result
    sf.write(output_path, result, out_sr)


def _enhance_metricgan_in_chunks(
    enhance_model,
    audio,   # torch.Tensor (1, T) at 16 kHz
    *,
    progress_callback=None,
    progress_start: int = 50,
    progress_end: int = 88,
):
    """Run MetricGAN+ on long audio with overlap-and-add chunking.

    Uses the same ``_chunk_window`` / overlap-add pattern as DeepFilterNet.
    """
    import torch

    SR = 16_000
    total_samples = int(audio.shape[-1])
    total_seconds = total_samples / SR

    if total_seconds <= _METRICGAN_CHUNK_THRESHOLD_SECONDS:
        if progress_callback:
            progress_callback(progress_start, "Enhancing speech (noise suppression)...")
        return enhance_model.enhance_batch(audio, lengths=torch.tensor([1.0])).detach().cpu()

    chunk_samples = SR * _METRICGAN_CHUNK_SECONDS
    overlap_samples = int(SR * _METRICGAN_OVERLAP_SECONDS)
    overlap_samples = min(overlap_samples, chunk_samples // 4)
    step_samples = max(1, chunk_samples - overlap_samples)

    segs: list[tuple[int, int]] = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        segs.append((start, end))
        if end >= total_samples:
            break
        start += step_samples

    num_segs = len(segs)
    output = torch.zeros_like(audio, dtype=torch.float32)
    weights = torch.zeros(total_samples, dtype=torch.float32)

    for idx, (seg_start, seg_end) in enumerate(segs, start=1):
        if progress_callback:
            pct = progress_start + int(
                ((idx - 1) / max(1, num_segs))
                * max(1, progress_end - progress_start)
            )
            progress_callback(pct, f"Enhancing chunk {idx}/{num_segs}...")

        chunk = audio[:, seg_start:seg_end]
        enhanced_chunk = (
            enhance_model.enhance_batch(chunk, lengths=torch.tensor([1.0]))
            .detach()
            .cpu()
        )
        enhanced_chunk = enhanced_chunk[..., : seg_end - seg_start].to(dtype=torch.float32)

        window = _chunk_window(
            seg_end - seg_start,
            fade_in=idx > 1,
            fade_out=idx < num_segs,
            overlap_samples=overlap_samples,
        )
        output[:, seg_start:seg_end] += enhanced_chunk * window.unsqueeze(0)
        weights[seg_start:seg_end] += window

        del enhanced_chunk, chunk
        _clear_torch_resources()

    weights = weights.clamp_min(1e-6)
    return (output / weights.unsqueeze(0)).cpu()


# ── ClearVoice Worker ──

class ClearVoiceWorker(QThread):
    """Worker thread for speech enhancement using ClearVoice models."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        model_name: str = "MossFormer2_SE_48K",
        remove_reverb: bool = True,
        dereverb_segment_size: int = _DEREVERB_SEPARATOR_PARAMS["segment_size"],
        dereverb_overlap: int = _DEREVERB_SEPARATOR_PARAMS["overlap"],
        dereverb_batch_size: int = _DEREVERB_SEPARATOR_PARAMS["batch_size"],
        dereverb_pitch_shift: int = _DEREVERB_SEPARATOR_PARAMS["pitch_shift"],
        auto_level: bool = True,
        match_input_loudness: bool = True,
        output_gain_db: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = model_name if model_name in CLEARVOICE_MODEL_SAMPLE_RATES else "MossFormer2_SE_48K"
        self.remove_reverb = remove_reverb
        self.dereverb_segment_size = dereverb_segment_size
        self.dereverb_overlap = dereverb_overlap
        self.dereverb_batch_size = dereverb_batch_size
        self.dereverb_pitch_shift = dereverb_pitch_shift
        self.auto_level = auto_level
        self.match_input_loudness = match_input_loudness
        self.output_gain_db = output_gain_db

    def run(self):
        try:
            import gc
            import tempfile

            from clearvoice import ClearVoice

            os.makedirs(self.output_dir, exist_ok=True)
            input_stem = Path(self.input_path).stem
            target_sample_rate = CLEARVOICE_MODEL_SAMPLE_RATES[self.model_name]

            self.progress.emit(5, "Preparing audio for ClearVoice...")
            with tempfile.TemporaryDirectory(prefix="ai-audio-toolkit-clearvoice-") as tmp_dir:
                prepared_input, metadata = _prepare_audio_for_model(
                    self.input_path,
                    tmp_dir,
                    target_sample_rate=target_sample_rate,
                    mono=True,
                    stem_suffix="clearvoice_input",
                )

                self.progress.emit(
                    12,
                    f"Loading ClearVoice model ({self.model_name}) (first run downloads checkpoints)...",
                )
                with _temporary_working_directory(_clearvoice_cache_dir()):
                    enhancer = ClearVoice(task="speech_enhancement", model_names=[self.model_name])

                    self.progress.emit(
                        42,
                        f"Enhancing speech with ClearVoice ({target_sample_rate // 1000} kHz model)...",
                    )
                    enhanced = enhancer(prepared_input, online_write=False)

                output_target_sr = _clearvoice_output_sample_rate(
                    self.model_name,
                    metadata["original_sample_rate"],
                )
                current_output = Path(self.output_dir) / (
                    f"{input_stem}_clearvoice_tmp.wav" if self.remove_reverb else f"{input_stem}_clearvoice.wav"
                )

                self.progress.emit(78, "Saving ClearVoice output...")
                _write_mono_float_wav(
                    current_output,
                    enhanced,
                    target_sample_rate,
                    target_sample_rate=output_target_sr,
                )

                if self.remove_reverb:
                    current_output = Path(
                        _run_dereverb_stage(
                            str(current_output),
                            self.output_dir,
                            Path(self.output_dir) / f"{input_stem}_clearvoice.wav",
                            mdxc_params={
                                "segment_size": self.dereverb_segment_size,
                                "overlap": self.dereverb_overlap,
                                "batch_size": self.dereverb_batch_size,
                                "pitch_shift": self.dereverb_pitch_shift,
                            },
                            progress_callback=self.progress.emit,
                            progress_start=80,
                            progress_end=92,
                        )
                    )
                    tmp_output = Path(self.output_dir) / f"{input_stem}_clearvoice_tmp.wav"
                    if tmp_output.is_file():
                        try:
                            tmp_output.unlink()
                        except OSError:
                            pass

                _apply_saved_audio_leveling(
                    self.input_path,
                    str(current_output),
                    auto_level=self.auto_level,
                    match_input_loudness=self.match_input_loudness,
                    output_gain_db=self.output_gain_db,
                    progress_callback=self.progress.emit,
                    progress_pct=94,
                )

            self.progress.emit(100, "ClearVoice enhancement complete!")
            self.finished.emit([str(current_output)])

            del enhanced, enhancer
            gc.collect()
            _clear_torch_resources()
        except Exception as e:
            import traceback

            log.error(traceback.format_exc())
            self.error.emit(str(e))


# ── MDX-Net Worker ──

class MdxNetWorker(QThread):
    """Worker thread for speech-forward cleanup using MDX-Net vocal isolation."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        model_name: str = "Kim Vocal 2",
        enable_denoise: bool = True,
        mdx_segment_size: int = _MDXNET_SEPARATOR_PARAMS["segment_size"],
        mdx_overlap: float = _MDXNET_SEPARATOR_PARAMS["overlap"],
        mdx_batch_size: int = _MDXNET_SEPARATOR_PARAMS["batch_size"],
        remove_reverb: bool = True,
        dereverb_segment_size: int = _DEREVERB_SEPARATOR_PARAMS["segment_size"],
        dereverb_overlap: int = _DEREVERB_SEPARATOR_PARAMS["overlap"],
        dereverb_batch_size: int = _DEREVERB_SEPARATOR_PARAMS["batch_size"],
        dereverb_pitch_shift: int = _DEREVERB_SEPARATOR_PARAMS["pitch_shift"],
        auto_level: bool = True,
        match_input_loudness: bool = True,
        output_gain_db: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = (
            model_name if model_name in MDXNET_ENHANCEMENT_PRESETS else "Kim Vocal 2"
        )
        self.enable_denoise = enable_denoise
        self.mdx_segment_size = mdx_segment_size
        self.mdx_overlap = mdx_overlap
        self.mdx_batch_size = mdx_batch_size
        self.remove_reverb = remove_reverb
        self.dereverb_segment_size = dereverb_segment_size
        self.dereverb_overlap = dereverb_overlap
        self.dereverb_batch_size = dereverb_batch_size
        self.dereverb_pitch_shift = dereverb_pitch_shift
        self.auto_level = auto_level
        self.match_input_loudness = match_input_loudness
        self.output_gain_db = output_gain_db

    def run(self):
        try:
            import gc

            from .separator_backend import get_preset_by_name

            preset = get_preset_by_name(self.model_name)
            if preset is None:
                raise RuntimeError(f"Unknown MDX-Net preset: {self.model_name}")

            os.makedirs(self.output_dir, exist_ok=True)
            input_stem = Path(self.input_path).stem
            separator_params = {
                "mdx_params": _normalize_mdx_separator_params(
                    segment_size=self.mdx_segment_size,
                    overlap=self.mdx_overlap,
                    batch_size=self.mdx_batch_size,
                    enable_denoise=self.enable_denoise,
                )
            }

            resolved_outputs = _run_separator_stage(
                self.input_path,
                self.output_dir,
                model_filename=preset.model_filename,
                separator_params=separator_params,
                progress_callback=self.progress.emit,
                progress_start=6,
                progress_end=74,
                loading_message=f"Loading MDX-Net model ({preset.name})...",
                processing_message="Isolating speech/vocals with MDX-Net...",
                complete_message="MDX-Net stem extraction complete.",
            )

            voice_source = _pick_separator_output(
                resolved_outputs,
                _MDXNET_PRIMARY_STEM_TOKENS + tuple(preset.stems[:1]),
            )
            if voice_source is None:
                raise RuntimeError("MDX-Net did not produce a speaker/voice stem.")

            background_candidates = [candidate for candidate in resolved_outputs if candidate != voice_source]
            background_source = _pick_separator_output(
                background_candidates,
                tuple(preset.stems[1:]) + _MDXNET_BACKGROUND_STEM_TOKENS,
            )

            speaker_output = Path(self.output_dir) / (
                f"{input_stem}_mdxnet_speaker_voice_tmp.wav"
                if self.remove_reverb
                else f"{input_stem}_mdxnet_speaker_voice.wav"
            )
            speaker_output = Path(_replace_file(voice_source, speaker_output))

            output_files: list[str] = []
            if background_source is not None:
                background_output = Path(self.output_dir) / f"{input_stem}_mdxnet_background_bleed_room.wav"
                background_output = Path(_replace_file(background_source, background_output))
                output_files.append(str(background_output))

            for candidate in resolved_outputs:
                if candidate in {voice_source, background_source}:
                    continue
                if candidate.is_file():
                    try:
                        candidate.unlink()
                    except OSError:
                        pass

            if self.remove_reverb:
                speaker_output = Path(
                    _run_dereverb_stage(
                        str(speaker_output),
                        self.output_dir,
                        Path(self.output_dir) / f"{input_stem}_mdxnet_speaker_voice.wav",
                        mdxc_params={
                            "segment_size": self.dereverb_segment_size,
                            "overlap": self.dereverb_overlap,
                            "batch_size": self.dereverb_batch_size,
                            "pitch_shift": self.dereverb_pitch_shift,
                        },
                        progress_callback=self.progress.emit,
                        progress_start=76,
                        progress_end=92,
                    )
                )
                tmp_output = Path(self.output_dir) / f"{input_stem}_mdxnet_speaker_voice_tmp.wav"
                if tmp_output.is_file():
                    try:
                        tmp_output.unlink()
                    except OSError:
                        pass

            _apply_saved_audio_leveling(
                self.input_path,
                str(speaker_output),
                auto_level=self.auto_level,
                match_input_loudness=self.match_input_loudness,
                output_gain_db=self.output_gain_db,
                progress_callback=self.progress.emit,
                progress_pct=94,
            )

            output_files.insert(0, str(speaker_output))
            self.progress.emit(100, "MDX-Net enhancement complete!")
            self.finished.emit(output_files)

            gc.collect()
            _clear_torch_resources()
        except Exception as e:
            import traceback

            log.error(traceback.format_exc())
            self.error.emit(str(e))


# ── VoiceFixer Worker ──

class VoiceFixerWorker(QThread):
    """Worker thread for all-in-one speech restoration using VoiceFixer.

    Handles noise removal, dereverberation, declipping, and audio
    super-resolution (up to 44.1 kHz) in a single model pass.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        mode: int = 0,
        auto_level: bool = True,
        match_input_loudness: bool = True,
        output_gain_db: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.mode = mode if int(mode) in VOICEFIXER_SUPPORTED_MODES else 0
        self.auto_level = auto_level
        self.match_input_loudness = match_input_loudness
        self.output_gain_db = output_gain_db

    def run(self):
        try:
            import gc
            import torch

            # ── Auto-install voicefixer if not present ──
            try:
                from voicefixer import VoiceFixer
            except ImportError:
                _pip_install("voicefixer", progress_callback=self.progress.emit,
                             label="voicefixer")
                # Invalidate cached check so the next run detects it
                global _voicefixer_checked
                _voicefixer_checked = False
                from voicefixer import VoiceFixer  # re-import after install

            self.progress.emit(
                8,
                f"Loading VoiceFixer model (first run downloads ~1 GB, mode {self.mode})...",
            )
            vf = VoiceFixer()
            cuda = torch.cuda.is_available()

            os.makedirs(self.output_dir, exist_ok=True)
            input_stem = Path(self.input_path).stem
            output_path = str(Path(self.output_dir) / f"{input_stem}_voicefixer.wav")

            _restore_voicefixer_in_chunks(
                vf,
                self.input_path,
                output_path,
                cuda=cuda,
                mode=self.mode,
                progress_callback=self.progress.emit,
                progress_start=20,
                progress_end=88,
            )

            _apply_saved_audio_leveling(
                self.input_path,
                output_path,
                auto_level=self.auto_level,
                match_input_loudness=self.match_input_loudness,
                output_gain_db=self.output_gain_db,
                progress_callback=self.progress.emit,
                progress_pct=92,
            )
            self.progress.emit(100, "VoiceFixer complete!")
            self.finished.emit([output_path])

            del vf
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            self.error.emit(str(e))


# ── MetricGAN+ Worker ──

class MetricGANWorker(QThread):
    """Worker thread for noise suppression using SpeechBrain MetricGAN+.

    State-of-the-art noise suppressor (PESQ 3.15 on VoiceBank+DEMAND).
    Handles noise only — pair with Studio Sound's reverb stage for full cleanup.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        auto_level: bool = True,
        match_input_loudness: bool = True,
        output_gain_db: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.auto_level = auto_level
        self.match_input_loudness = match_input_loudness
        self.output_gain_db = output_gain_db

    def run(self):
        try:
            import gc
            import tempfile
            import soundfile as sf
            import torch
            import torchaudio

            # ── Stub optional SpeechBrain integrations (k2, etc.) ──
            _install_speechbrain_stubs()

            # ── Auto-install speechbrain if not present ──
            try:
                from speechbrain.inference.enhancement import SpectralMaskEnhancement
            except ImportError:
                _pip_install("speechbrain", progress_callback=self.progress.emit,
                             label="speechbrain")
                global _metricgan_checked
                _metricgan_checked = False
                _install_speechbrain_stubs()
                from speechbrain.inference.enhancement import SpectralMaskEnhancement

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.progress.emit(8, "Loading MetricGAN+ model (first run downloads ~100 MB)...")

            enhance_model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir=str(Path(_hf_cache_dir()) / "metricgan-plus-voicebank"),
                run_opts={"device": device},
            )

            self.progress.emit(30, "Loading audio...")
            with tempfile.TemporaryDirectory(prefix="ai-audio-toolkit-metricgan-") as tmp_dir:
                prepared_input, metadata = _prepare_audio_for_model(
                    self.input_path,
                    tmp_dir,
                    target_sample_rate=16_000,
                    mono=True,
                    stem_suffix="metricgan_input",
                )
                wav_np, _ = sf.read(prepared_input, always_2d=True)
                wav = torch.from_numpy(wav_np.T.copy()).to(dtype=torch.float32)
                orig_sr = int(metadata["original_sample_rate"])

                enhanced = _enhance_metricgan_in_chunks(
                    enhance_model,
                    wav,
                    progress_callback=self.progress.emit,
                    progress_start=50,
                    progress_end=88,
                )

                os.makedirs(self.output_dir, exist_ok=True)
                input_stem = Path(self.input_path).stem
                output_path = str(Path(self.output_dir) / f"{input_stem}_metricgan.wav")

                out_wav = _coerce_audio_batch_tensor(enhanced)
                if orig_sr != 16_000:
                    out_wav = torchaudio.functional.resample(out_wav, 16_000, orig_sr)
                torchaudio.save(output_path, out_wav, orig_sr)

            _apply_saved_audio_leveling(
                self.input_path,
                output_path,
                auto_level=self.auto_level,
                match_input_loudness=self.match_input_loudness,
                output_gain_db=self.output_gain_db,
                progress_callback=self.progress.emit,
                progress_pct=92,
            )
            self.progress.emit(100, "MetricGAN+ complete!")
            self.finished.emit([output_path])

            del enhance_model, enhanced, out_wav
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            self.error.emit(str(e))


# ── Enhancement Worker ──

class EnhanceWorker(QThread):
    """Worker thread for audio enhancement using Resemble-Enhance."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)  # output file paths
    error = pyqtSignal(str)

    def __init__(self, input_path: str, output_dir: str,
                 mode: str = "enhance",
                 nfe: int = 64,
                 solver: str = "midpoint",
                 prior_temperature: float = 0.5,
                 cfm_guidance: float = 0.9,
                 denoise_before: bool = False,
                 auto_level: bool = True,
                 match_input_loudness: bool = True,
                 output_gain_db: float = 0.0,
                 parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.mode = mode  # "enhance", "denoise"
        self.nfe = nfe
        self.solver = solver
        self.prior_temperature = prior_temperature
        self.cfm_guidance = cfm_guidance
        self.denoise_before = denoise_before
        self.auto_level = auto_level
        self.match_input_loudness = match_input_loudness
        self.output_gain_db = output_gain_db

    def run(self):
        try:
            import torch
            import torchaudio

            _install_deepspeed_stub()
            _patch_download()
            from resemble_enhance.enhancer.inference import denoise, enhance

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load audio
            self.progress.emit(5, "Loading audio...")
            wav, sr = torchaudio.load(self.input_path)
            # Mix to mono (resemble-enhance expects mono)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)
            log.info("Loaded audio: %d samples @ %dHz", wav.shape[-1], sr)

            # Download model if needed
            self.progress.emit(10, "Loading Resemble-Enhance model (first run downloads ~1.5GB)...")

            os.makedirs(self.output_dir, exist_ok=True)
            input_stem = Path(self.input_path).stem

            # Ensure sr is a plain Python int (not numpy/tensor scalar)
            sr = int(sr)

            if self.mode == "denoise":
                self.progress.emit(30, "Denoising audio...")
                out_wav, out_sr = denoise(wav, sr, device)
                out_sr = int(out_sr)

                output_path = str(
                    Path(self.output_dir) / f"{input_stem}_denoised.wav"
                )
                out_tensor = out_wav.cpu()
                if out_tensor.dim() == 1:
                    out_tensor = out_tensor.unsqueeze(0)
                torchaudio.save(output_path, out_tensor, out_sr)
                _apply_saved_audio_leveling(
                    self.input_path,
                    output_path,
                    auto_level=self.auto_level,
                    match_input_loudness=self.match_input_loudness,
                    output_gain_db=self.output_gain_db,
                    progress_callback=self.progress.emit,
                    progress_pct=94,
                )
                log.info("Saved denoised: %s", output_path)

                self.progress.emit(100, "Denoising complete!")
                self.finished.emit([output_path])

            else:
                # Full enhancement
                if self.denoise_before:
                    self.progress.emit(20, "Denoising before enhancement...")
                    wav, sr = denoise(wav, sr, device)
                    sr = int(sr)
                    wav = wav.cpu()
                    log.info("Pre-denoise complete")

                self.progress.emit(40, f"Enhancing audio (NFE={self.nfe}, solver={self.solver})...")

                out_wav, out_sr = enhance(
                    wav, sr, device,
                    nfe=self.nfe,
                    solver=self.solver,
                    lambd=self.cfm_guidance,
                    tau=self.prior_temperature,
                )
                out_sr = int(out_sr)

                output_path = str(
                    Path(self.output_dir) / f"{input_stem}_enhanced.wav"
                )
                out_tensor = out_wav.cpu()
                if out_tensor.dim() == 1:
                    out_tensor = out_tensor.unsqueeze(0)
                torchaudio.save(output_path, out_tensor, out_sr)
                _apply_saved_audio_leveling(
                    self.input_path,
                    output_path,
                    auto_level=self.auto_level,
                    match_input_loudness=self.match_input_loudness,
                    output_gain_db=self.output_gain_db,
                    progress_callback=self.progress.emit,
                    progress_pct=94,
                )
                log.info("Saved enhanced: %s", output_path)

                self.progress.emit(100, "Enhancement complete!")
                self.finished.emit([output_path])

            # Cleanup
            del out_wav, wav
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            self.error.emit(str(e))
