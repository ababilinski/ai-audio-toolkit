"""Legacy cx_Freeze build script for ai-audio-toolkit.

This remains in the repo for diagnostics only. The supported Windows release
build is `build_bundle.py`, which keeps the GPU/ML runtime unfrozen.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

import librosa
import matplotlib
from cx_Freeze import Executable, setup
from cx_Freeze.freezer import AddIcon, UpdateCheckSum

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build" / "cx_freeze"
ASSETS_DIR = ROOT / "assets"
ICON_PATH = ASSETS_DIR / "app_icon.ico"
FFMPEG_ROOTS = [
    Path.home() / ".ai-audio-toolkit" / "ffmpeg-shared" / "ffmpeg-8.1-full_build-shared",
    Path.home() / ".audio-editor" / "ffmpeg-shared" / "ffmpeg-8.1-full_build-shared",
]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


packages = [
    "audio_editor",
    "PyQt6",
    "numpy",
    "scipy",
    "soundfile",
    "sounddevice",
    "librosa",
    "matplotlib",
    "pydub",
    "audio_separator",
    "torch",
    "torchaudio",
    "transformers",
    "huggingface_hub",
]

optional_packages = [
    name
    for name in [
        "sam_audio",
        "resemble_enhance",
        "cv2",
        "sam2",
        "sam3",
        "df",
        "libdf",
        "appdirs",
        "loguru",
    ]
    if _has_module(name)
]

include_files: list[tuple[str, str]] = [
    (str(ASSETS_DIR), "assets"),
    (matplotlib.get_data_path(), "matplotlib/mpl-data"),
]

LIBROSA_DIR = Path(librosa.__file__).resolve().parent
for relative_stub in [
    "__init__.pyi",
    "core/__init__.pyi",
    "feature/__init__.pyi",
    "util/__init__.pyi",
]:
    stub_path = LIBROSA_DIR / relative_stub
    if stub_path.is_file():
        include_files.append((str(stub_path), f"lib/librosa/{relative_stub}"))

DEEPFILTER_DLLS_DIR = Path(sys.prefix) / "Lib" / "site-packages" / "DeepFilterLib.libs"
if DEEPFILTER_DLLS_DIR.is_dir():
    include_files.append((str(DEEPFILTER_DLLS_DIR), "DeepFilterLib.libs"))

FFMPEG_ROOT = next((root for root in FFMPEG_ROOTS if (root / "bin").is_dir()), None)
if FFMPEG_ROOT is not None:
    include_files.append((str(FFMPEG_ROOT / "bin"), "ffmpeg/bin"))
    if (FFMPEG_ROOT / "LICENSE").is_file():
        include_files.append((str(FFMPEG_ROOT / "LICENSE"), "ffmpeg/LICENSE"))
    if (FFMPEG_ROOT / "README.txt").is_file():
        include_files.append((str(FFMPEG_ROOT / "README.txt"), "ffmpeg/README.txt"))

build_exe_options = {
    "build_exe": str(BUILD_DIR),
    "packages": packages + optional_packages,
    "includes": [
        "PyQt6.QtMultimedia",
        "PyQt6.QtMultimediaWidgets",
        "PyQt6.QtGui",
        "PyQt6.QtCore",
        "PyQt6.QtWidgets",
    ],
    "include_files": include_files,
    "include_msvcr": True,
    "excludes": [
        "tkinter",
        "test",
        "unittest",
        "pydoc_data",
        "setuptools",
        "pip",
        "wheel",
    ],
    "zip_include_packages": [],
    "zip_exclude_packages": ["*"],
    "optimize": 0,
    "silent_level": 1,
}

base = "Win32GUI" if sys.platform == "win32" else None

executables = [
    Executable(
        script=str(ROOT / "run.py"),
        target_name="ai-audio-toolkit.exe",
        base=base,
        icon=None,
    ),
]


def _stamp_exe_icon() -> None:
    if os.environ.get("AUDIO_EDITOR_SKIP_EXE_ICON") or not ICON_PATH.is_file():
        return
    target_exe = BUILD_DIR / "ai-audio-toolkit.exe"
    if not target_exe.is_file():
        return
    AddIcon(target_exe, ICON_PATH)
    UpdateCheckSum(target_exe)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("build_exe")

    try:
        setup(
            name="ai-audio-toolkit",
            version="1.0.0",
            description="ai-audio-toolkit with AI-powered stem separation",
            options={"build_exe": build_exe_options},
            executables=executables,
        )
        _stamp_exe_icon()
    except Exception:
        traceback.print_exc()
        raise
