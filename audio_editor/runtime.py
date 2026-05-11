"""Runtime helpers for source, frozen, and bundled application builds."""
from __future__ import annotations

import ctypes
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

from .branding import APP_DISPLAY_NAME, LEGACY_DOT_DIR

APP_USER_MODEL_ID = f"{APP_DISPLAY_NAME}.{APP_DISPLAY_NAME}"
_HOME_FFMPEG_BINS = (
    Path.home()
    / f".{APP_DISPLAY_NAME}"
    / "ffmpeg-shared"
    / "ffmpeg-8.1-full_build-shared"
    / "bin",
    Path.home()
    / LEGACY_DOT_DIR
    / "ffmpeg-shared"
    / "ffmpeg-8.1-full_build-shared"
    / "bin",
)
_BUNDLE_ENVS = ("AI_AUDIO_TOOLKIT_BUNDLE_ROOT", "AUDIO_EDITOR_BUNDLE_ROOT")
_TORCH_RUNTIME_PRELOADED = False


def is_frozen() -> bool:
    """Return True when running from a cx_Freeze or PyInstaller build."""
    return bool(getattr(sys, "frozen", False)) or hasattr(sys, "_MEIPASS")


def package_root() -> Path:
    """Return the package directory for source runs."""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """Return the repository root for source runs."""
    return package_root().parent


def bundle_root() -> Path | None:
    """Return the explicit launcher-provided bundle root, when present."""
    override = next(
        (os.environ.get(name, "").strip() for name in _BUNDLE_ENVS if os.environ.get(name, "").strip()),
        "",
    )
    if not override:
        return None
    try:
        path = Path(override).expanduser().resolve()
    except OSError:
        return None
    return path if path.exists() else None


def binary_root() -> Path:
    """Return the executable/bundle directory or project root."""
    root = bundle_root()
    if root is not None:
        return root
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return project_root()


def _load_settings():
    try:
        from .settings import get_app_settings
    except Exception:
        return None
    try:
        return get_app_settings()
    except Exception:
        return None


def _candidate_asset_roots() -> list[Path]:
    root = binary_root()
    candidates = [
        root / "assets",
        root / "resources" / "assets",
        project_root() / "assets",
    ]
    return [path for path in candidates if path not in ()]


def get_asset_path(name: str) -> Path | None:
    """Locate an asset file by name."""
    for root in _candidate_asset_roots():
        path = root / name
        if path.is_file():
            return path
    return None


def get_icon_path() -> Path | None:
    """Return the application ICO path when present."""
    return get_asset_path("app_icon.ico")


def preload_torch_runtime(*, use_settings: bool = False) -> None:
    """Load torch before Qt on Windows so CUDA DLLs initialize reliably."""
    global _TORCH_RUNTIME_PRELOADED

    if _TORCH_RUNTIME_PRELOADED:
        return
    if sys.platform != "win32":
        _TORCH_RUNTIME_PRELOADED = True
        return
    preload_torch = os.environ.get(
        "AI_AUDIO_TOOLKIT_PRELOAD_TORCH",
        os.environ.get("AUDIO_EDITOR_PRELOAD_TORCH", "1"),
    )
    if preload_torch != "1":
        _TORCH_RUNTIME_PRELOADED = True
        return

    ensure_runtime_dll_directories(use_settings=use_settings)
    warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*triton.*")

    try:
        logging.getLogger("xformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
        logging.getLogger("sam_audio").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import torch  # noqa: F401
    except Exception:
        return

    _TORCH_RUNTIME_PRELOADED = True


def _ffmpeg_override_candidates(use_settings: bool) -> list[Path]:
    if not use_settings:
        return []
    settings = _load_settings()
    if settings is None:
        return []

    override = settings.ffmpeg_override()
    if not override:
        return []

    path = Path(override)
    if path.is_file():
        return [path.parent]
    if path.is_dir():
        candidates = [path]
        if (path / "bin").is_dir():
            candidates.insert(0, path / "bin")
        return candidates
    return []


def _candidate_ffmpeg_bin_dirs(*, use_settings: bool = True) -> list[Path]:
    root = binary_root()
    bundled_candidates = [
        root / "ffmpeg" / "bin",
        root / "resources" / "ffmpeg" / "bin",
        root / "python" / "ffmpeg" / "bin",
    ]
    configured_candidates = _ffmpeg_override_candidates(use_settings)
    if bundle_root() is not None or is_frozen():
        candidates = [*bundled_candidates, *configured_candidates, *_HOME_FFMPEG_BINS]
    else:
        candidates = [*configured_candidates, *bundled_candidates, *_HOME_FFMPEG_BINS]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _candidate_native_dll_dirs(*, use_settings: bool = True) -> list[Path]:
    root = binary_root()
    site_packages = root / "python" / "Lib" / "site-packages"

    extra_paths: list[Path] = []
    if use_settings:
        settings = _load_settings()
        if settings is not None:
            extra_paths = [Path(path) for path in settings.extra_runtime_paths()]

    candidates = [
        *extra_paths,
        root / "python",
        root / "python" / "DLLs",
        root / "python" / "Lib" / "site-packages",
        site_packages / "torch" / "lib",
        site_packages / "torch.libs",
        site_packages / "torchcodec",
        site_packages / "onnxruntime" / "capi",
        site_packages / "DeepFilterLib.libs",
        site_packages / "av.libs",
        root / "ffmpeg" / "bin",
        # cx_Freeze layout (lib/ subdirectory)
        root / "lib",
        root / "lib" / "torch" / "lib",
        root / "lib" / "onnxruntime" / "capi",
        root / "DeepFilterLib.libs",
        root / "lib" / "av.libs",
        # PyInstaller one-dir layout (flat alongside EXE)
        root / "torch" / "lib",
        root / "torch.libs",
        root / "onnxruntime" / "capi",
        root,
    ]

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def get_ffmpeg_bin_dir(*, use_settings: bool = True) -> Path | None:
    """Locate a bundled or configured FFmpeg bin directory."""
    for path in _candidate_ffmpeg_bin_dirs(use_settings=use_settings):
        if (path / "ffmpeg.exe").is_file() or (path / "ffmpeg").is_file():
            return path
    which_path = shutil.which("ffmpeg")
    if which_path:
        return Path(which_path).resolve().parent
    return None


def ensure_runtime_dll_directories(*, use_settings: bool = True) -> list[Path]:
    """Expose bundled native library directories to Windows DLL search."""
    if sys.platform != "win32":
        return []

    added: list[Path] = []
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []

    for directory in _candidate_native_dll_dirs(use_settings=use_settings):
        if not directory.is_dir():
            continue

        directory_str = str(directory)
        if directory_str not in path_entries:
            path_entries.insert(0, directory_str)
        try:
            os.add_dll_directory(directory_str)
        except (AttributeError, FileNotFoundError, OSError):
            pass
        added.append(directory)

    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries)
    return added


def get_ffmpeg_executable(*, use_settings: bool = True) -> str | None:
    """Locate the ffmpeg executable."""
    ffmpeg_dir = get_ffmpeg_bin_dir(use_settings=use_settings)
    if ffmpeg_dir is not None:
        exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        candidate = ffmpeg_dir / exe_name
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


def require_ffmpeg_executable() -> str:
    """Return an ffmpeg executable path or raise a clear error."""
    ffmpeg = get_ffmpeg_executable()
    if ffmpeg:
        return ffmpeg
    raise FileNotFoundError(
        "FFmpeg is required for video features but was not found. "
        "Use the bundled runtime, configure an FFmpeg path in Settings, or install FFmpeg and add it to PATH."
    )


def get_ffprobe_executable(*, use_settings: bool = True) -> str | None:
    """Locate the ffprobe executable."""
    ffmpeg_dir = get_ffmpeg_bin_dir(use_settings=use_settings)
    if ffmpeg_dir is not None:
        exe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
        candidate = ffmpeg_dir / exe_name
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffprobe")


def ffmpeg_command(*args: str) -> list[str]:
    """Build an ffmpeg command that works in source and bundled modes."""
    return [require_ffmpeg_executable(), *args]


def ensure_ffmpeg_environment(*, use_settings: bool = True) -> Path | None:
    """Expose bundled/configured FFmpeg to subprocesses and torchcodec."""
    ffmpeg_dir = get_ffmpeg_bin_dir(use_settings=use_settings)
    if ffmpeg_dir is None:
        return None

    current_path = os.environ.get("PATH", "")
    ffmpeg_dir_str = str(ffmpeg_dir)
    path_entries = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir_str not in path_entries:
        os.environ["PATH"] = os.pathsep.join([ffmpeg_dir_str, *path_entries]) if current_path else ffmpeg_dir_str

    if sys.platform == "win32":
        try:
            os.add_dll_directory(ffmpeg_dir_str)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    return ffmpeg_dir


def ensure_windows_app_id() -> None:
    """Set an explicit AppUserModelID so the taskbar uses the app icon."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass
