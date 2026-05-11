"""Legacy frozen PyInstaller build script for ai-audio-toolkit.

Usage:
    .venv\\Scripts\\python.exe build_pyinstaller.py

Output:
    build\\pyinstaller\\ai-audio-toolkit\\ai-audio-toolkit.exe

This is kept for diagnostics. The supported Windows release build is
`build_bundle.py`, which launches a bundled CPython runtime instead of
freezing the ML stack.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "build" / "pyinstaller"
WORK_DIR = ROOT / "build" / "pyinstaller_work"
SPEC = ROOT / "audio_editor.spec"


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found — installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            stdout=subprocess.DEVNULL,
        )


def _run_pyinstaller() -> None:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(SPEC),
        "--distpath", str(DIST_DIR),
        "--workpath", str(WORK_DIR),
        "--noconfirm",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    _ensure_pyinstaller()
    _run_pyinstaller()

    output_exe = DIST_DIR / "ai-audio-toolkit" / "ai-audio-toolkit.exe"
    if output_exe.is_file():
        size_mb = output_exe.stat().st_size / 1_048_576
        print(f"\nBuild successful!")
        print(f"  EXE : {output_exe}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"\nTo run: {output_exe}")
    else:
        print("\nBuild may have failed — EXE not found at expected path.")
        print(f"  Expected: {output_exe}")
        sys.exit(1)


if __name__ == "__main__":
    main()
