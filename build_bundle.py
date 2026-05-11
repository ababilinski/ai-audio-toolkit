"""Build a natural Windows app bundle with a launcher EXE and bundled CPython runtime.

Usage:
    .venv\\Scripts\\python.exe build_bundle.py

Output:
    build\\windows_bundle\\ai-audio-toolkit\\ai-audio-toolkit.exe
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
BUILD_ROOT = ROOT / "build" / "windows_bundle"
APP_DIR = BUILD_ROOT / "ai-audio-toolkit"
PYTHON_DIR = APP_DIR / "python"
WORK_ROOT = ROOT / "build" / "windows_bundle_work"
LAUNCHER_SCRIPT = ROOT / "scripts" / "windows_launcher.py"
LAUNCHER_DIST = WORK_ROOT / "launcher_dist"
LAUNCHER_WORK = WORK_ROOT / "launcher_work"
LAUNCHER_SPEC = WORK_ROOT / "launcher_spec"


def _on_rm_error(func, path, exc_info):
    target = Path(path)
    try:
        target.chmod(stat.S_IWRITE)
    except OSError:
        pass
    func(path)


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_on_rm_error)


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found, installing it into the current environment...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            stdout=subprocess.DEVNULL,
        )


def _venv_base_home() -> Path:
    cfg_path = VENV_DIR / "pyvenv.cfg"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing virtualenv config: {cfg_path}")
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("home ="):
            home = Path(line.split("=", 1)[1].strip())
            if home.is_dir():
                return home
            break
    raise RuntimeError("Could not resolve the base CPython home from .venv/pyvenv.cfg")


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_python_runtime() -> None:
    base_home = _venv_base_home()
    site_packages_src = VENV_DIR / "Lib" / "site-packages"

    print(f"Copying CPython runtime from {base_home} ...")
    _copytree(base_home, PYTHON_DIR)

    print(f"Copying site-packages from {site_packages_src} ...")
    _copytree(site_packages_src, PYTHON_DIR / "Lib" / "site-packages")


def _copy_app_sources() -> None:
    print("Copying application sources and assets ...")
    _copytree(ROOT / "audio_editor", APP_DIR / "audio_editor")
    assets_dir = ROOT / "assets"
    if assets_dir.is_dir():
        _copytree(assets_dir, APP_DIR / "assets")


def _copy_ffmpeg() -> None:
    from audio_editor.runtime import get_ffmpeg_bin_dir

    ffmpeg_bin = get_ffmpeg_bin_dir()
    if ffmpeg_bin is None:
        raise FileNotFoundError(
            "FFmpeg was not found. Configure it in Settings or install a shared FFmpeg build before bundling."
        )
    print(f"Copying FFmpeg from {ffmpeg_bin} ...")
    _copytree(ffmpeg_bin, APP_DIR / "ffmpeg" / "bin")


def _prune_bundle(root: Path) -> None:
    print("Pruning bundle ...")
    removable_dirs = [
        root / "python" / "Lib" / "test",
        root / "python" / "Tools",
        root / "python" / "Scripts",
        root / "python" / "Lib" / "site-packages" / "resemble_enhance" / "model_repo" / ".git",
    ]
    for directory in removable_dirs:
        _safe_rmtree(directory)

    for path in list(root.rglob("*")):
        if path.is_dir() and path.name == "__pycache__":
            _safe_rmtree(path)
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".pdb", ".lib"}:
            try:
                path.unlink()
            except OSError:
                pass


def _build_launcher() -> None:
    _ensure_pyinstaller()
    icon_path = ROOT / "assets" / "app_icon.ico"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(LAUNCHER_SCRIPT),
        "--name",
        "ai-audio-toolkit",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(LAUNCHER_DIST),
        "--workpath",
        str(LAUNCHER_WORK),
        "--specpath",
        str(LAUNCHER_SPEC),
    ]
    if icon_path.is_file():
        cmd.extend(["--icon", str(icon_path)])

    print("Building launcher EXE ...")
    subprocess.check_call(cmd, cwd=ROOT)

    built_exe = LAUNCHER_DIST / "ai-audio-toolkit.exe"
    if not built_exe.is_file():
        raise FileNotFoundError(f"PyInstaller did not produce {built_exe}")
    shutil.copy2(built_exe, APP_DIR / "ai-audio-toolkit.exe")


def _prepare_dirs() -> None:
    _safe_rmtree(BUILD_ROOT)
    _safe_rmtree(WORK_ROOT)
    APP_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if sys.platform != "win32":
        print("build_bundle.py currently supports Windows only.")
        return 2
    if not VENV_DIR.is_dir():
        print("The .venv directory was not found. Build from the project root after creating the virtualenv.")
        return 2
    if not LAUNCHER_SCRIPT.is_file():
        print(f"Missing launcher script: {LAUNCHER_SCRIPT}")
        return 2

    print("Preparing output directories ...")
    _prepare_dirs()
    _copy_python_runtime()
    _copy_app_sources()
    _copy_ffmpeg()
    _prune_bundle(APP_DIR)
    _build_launcher()

    output_exe = APP_DIR / "ai-audio-toolkit.exe"
    if not output_exe.is_file():
        print(f"Build completed, but the launcher EXE was not found at {output_exe}")
        return 1

    print("\nBundle build successful.")
    print(f"Output: {output_exe}")
    print(f"Bundle root: {APP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
