"""Entry point for the ai-audio-toolkit application."""
import argparse
import os
import sys
import warnings

from .runtime import (
    ensure_ffmpeg_environment,
    ensure_runtime_dll_directories,
    ensure_windows_app_id,
    get_icon_path,
    preload_torch_runtime,
)
from .branding import APP_DISPLAY_NAME

# Suppress noisy warnings from torch/xformers on Windows
os.environ.setdefault("XFORMERS_MORE_DETAILS", "0")
os.environ.setdefault("TRITON_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")  # suppress abseil/glog W-level logs
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", message=".*triton.*")
ensure_runtime_dll_directories(use_settings=False)
preload_torch_runtime(use_settings=False)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import QApplication

try:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
except Exception:
    pass


def _apply_managed_runtime() -> None:
    """Apply the app-managed settings to the current process runtime."""
    from .settings import get_app_settings

    settings = get_app_settings()
    settings.apply_runtime_environment()
    ensure_runtime_dll_directories()
    ensure_ffmpeg_environment()


def create_application(argv: list[str] | None = None) -> QApplication:
    """Create and configure the QApplication instance."""
    _apply_managed_runtime()
    ensure_windows_app_id()
    ensure_runtime_dll_directories()
    ensure_ffmpeg_environment()

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationVersion("1.0.0")
    app.setFont(QFont("Segoe UI", 10))

    icon_path = get_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--self-test-mode",
        choices=[
            "startup",
            "availability",
            "sam",
            "enhance",
            "deepfilter",
            "clearvoice",
            "audio-separator",
            "voicefixer",
            "metricgan",
            "enhancement-regressions",
            "gpu-stack",
            "settings-roundtrip",
            "settings-model-scan",
            "separator-imports",
            "video",
            "separator",
            "suite",
        ],
        default="suite",
    )
    parser.add_argument("--self-test-workdir")
    parser.add_argument("--self-test-report")
    parser.add_argument("--self-test-model-filename", default="kuielab_a_drums.onnx")
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def main():
    args = _parse_args()
    _apply_managed_runtime()
    if args.self_test:
        from .selftest import main as selftest_main

        if not args.self_test_workdir:
            print("--self-test-workdir is required when running self-tests.")
            return 2
        return selftest_main([
            "--mode", args.self_test_mode,
            "--workdir", args.self_test_workdir,
            "--model-filename", args.self_test_model_filename,
            *(["--report", args.self_test_report] if args.self_test_report else []),
        ])

    app = create_application()

    from .main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
