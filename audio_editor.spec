# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ai-audio-toolkit.

Build with:
    .venv\\Scripts\\python.exe build_pyinstaller.py
or directly:
    .venv\\Scripts\\pyinstaller.exe audio_editor.spec --distpath build/pyinstaller --workpath build/pyinstaller_work --noconfirm
"""
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = Path(SPECPATH)


def _has(name):
    return importlib.util.find_spec(name) is not None


# ── Core ML packages — collect_all grabs DLLs, data files, and hidden imports ──
datas_torch,    binaries_torch,    hidden_torch    = collect_all("torch")
datas_ta,       binaries_ta,       hidden_ta       = collect_all("torchaudio")
datas_sep,      binaries_sep,      hidden_sep      = collect_all("audio_separator")
datas_lib,      binaries_lib,      hidden_lib      = collect_all("librosa")
datas_mpl,      binaries_mpl,      hidden_mpl      = collect_all("matplotlib")
datas_scipy,    binaries_scipy,    hidden_scipy    = collect_all("scipy")
datas_sf,       binaries_sf,       hidden_sf       = collect_all("soundfile")
datas_hf,       binaries_hf,       hidden_hf       = collect_all("huggingface_hub")
datas_tf,       binaries_tf,       hidden_tf       = collect_all("transformers")

all_datas    = (datas_torch + datas_ta + datas_sep + datas_lib + datas_mpl
                + datas_scipy + datas_sf + datas_hf + datas_tf)
all_binaries = (binaries_torch + binaries_ta + binaries_sep + binaries_lib + binaries_mpl
                + binaries_scipy + binaries_sf + binaries_hf + binaries_tf)
all_hidden   = (hidden_torch + hidden_ta + hidden_sep + hidden_lib + hidden_mpl
                + hidden_scipy + hidden_sf + hidden_hf + hidden_tf)

# ── Optional packages ──
for opt in ["sam_audio", "resemble_enhance", "cv2", "df", "libdf", "loguru",
            "voicefixer", "speechbrain"]:
    if _has(opt):
        d, b, h = collect_all(opt)
        all_datas    += d
        all_binaries += b
        all_hidden   += h

# ── App assets ──
all_datas += [("assets", "assets")]

# ── Bundled FFmpeg ──
_ffmpeg_roots = [
    Path.home() / ".ai-audio-toolkit" / "ffmpeg-shared" / "ffmpeg-8.1-full_build-shared",
    Path.home() / ".audio-editor" / "ffmpeg-shared" / "ffmpeg-8.1-full_build-shared",
]
_ffmpeg_root = next((root for root in _ffmpeg_roots if (root / "bin").is_dir()), _ffmpeg_roots[0])
if (_ffmpeg_root / "bin").is_dir():
    all_datas += [
        (str(_ffmpeg_root / "bin"), "ffmpeg/bin"),
    ]
    for extra in ["LICENSE", "README.txt"]:
        p = _ffmpeg_root / extra
        if p.is_file():
            all_datas += [(str(p), f"ffmpeg/{extra}")]

# ── DeepFilterLib native DLLs (if installed as a separate .libs wheel) ──
import sys as _sys
_df_libs = Path(_sys.prefix) / "Lib" / "site-packages" / "DeepFilterLib.libs"
if _df_libs.is_dir():
    all_datas += [(str(_df_libs), "DeepFilterLib.libs")]

# ── Hidden imports ──
all_hidden += collect_submodules("audio_editor")
all_hidden += [
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtGui",
    "PyQt6.QtCore",
    "PyQt6.QtWidgets",
    "sounddevice",
    "scipy.signal",
    "scipy.fft",
    "scipy.optimize",
    "scipy.interpolate",
    "scipy.linalg",
    "pydub",
    "numpy",
    "PIL",
    "PIL.Image",
    "onnxruntime",
    # VoiceFixer optional engine
    "voicefixer",
    # SpeechBrain MetricGAN+ optional engine
    "speechbrain",
    "speechbrain.inference",
    "speechbrain.inference.enhancement",
    "speechbrain.pretrained",
    "speechbrain.pretrained.interfaces",
]

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "rthooks" / "rthook_setup.py")],
    excludes=["tkinter", "pydoc_data"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ai-audio-toolkit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,         # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(ROOT / "assets" / "app_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ai-audio-toolkit",
)
