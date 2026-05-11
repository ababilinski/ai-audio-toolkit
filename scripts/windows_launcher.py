"""Launcher for the bundled Windows ai-audio-toolkit runtime."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _prepend_path(env: dict[str, str], path: Path) -> None:
    if not path.exists():
        return
    current = env.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    path_str = str(path)
    if path_str not in parts:
        env["PATH"] = os.pathsep.join([path_str, *parts]) if current else path_str


def _is_console_mode(args: list[str]) -> bool:
    console_flags = {
        "--self-test",
        "--self-test-mode",
        "--self-test-workdir",
        "--self-test-report",
        "--help",
        "-h",
    }
    return any(arg in console_flags for arg in args)


def _python_executable(root: Path, *, console_mode: bool) -> Path:
    python_dir = root / "python"
    preferred = python_dir / ("python.exe" if console_mode else "pythonw.exe")
    if preferred.is_file():
        return preferred
    fallback = python_dir / "python.exe"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Bundled Python executable was not found under {python_dir}")


def _runtime_paths(root: Path) -> list[Path]:
    site_packages = root / "python" / "Lib" / "site-packages"
    return [
        root / "python",
        root / "python" / "DLLs",
        site_packages,
        site_packages / "torch" / "lib",
        site_packages / "torch.libs",
        site_packages / "torchcodec",
        site_packages / "onnxruntime" / "capi",
        site_packages / "DeepFilterLib.libs",
        site_packages / "av.libs",
        root / "ffmpeg" / "bin",
    ]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = _bundle_root()
    python_home = root / "python"
    site_packages = python_home / "Lib" / "site-packages"
    console_mode = _is_console_mode(args)

    env = os.environ.copy()
    env["AI_AUDIO_TOOLKIT_BUNDLE_ROOT"] = str(root)
    env["AUDIO_EDITOR_BUNDLE_ROOT"] = str(root)
    env["PYTHONHOME"] = str(python_home)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(site_packages)])
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env.setdefault("AI_AUDIO_TOOLKIT_PRELOAD_TORCH", env.get("AUDIO_EDITOR_PRELOAD_TORCH", "1"))
    env.setdefault("AUDIO_EDITOR_PRELOAD_TORCH", env["AI_AUDIO_TOOLKIT_PRELOAD_TORCH"])

    for path in reversed(_runtime_paths(root)):
        _prepend_path(env, path)

    python_exe = _python_executable(root, console_mode=console_mode)
    command = [str(python_exe), "-m", "audio_editor.main", *args]
    completed = subprocess.run(command, cwd=root, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
