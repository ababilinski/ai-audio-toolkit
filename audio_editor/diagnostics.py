"""Runtime diagnostics for acceleration and external tool setup."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .runtime import get_ffmpeg_executable

if TYPE_CHECKING:
    from .settings import AppSettings


@dataclass(slots=True)
class AccelerationReport:
    """Summary of runtime acceleration availability."""

    mode: str
    reason_code: str
    summary: str
    details: str
    gpu_name: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    onnxruntime_version: str = ""
    ffmpeg_path: str = ""
    nvidia_smi_path: str = ""
    torch_error: str = ""
    onnxruntime_error: str = ""
    providers: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "reason_code": self.reason_code,
            "summary": self.summary,
            "details": self.details,
            "gpu_name": self.gpu_name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "onnxruntime_version": self.onnxruntime_version,
            "ffmpeg_path": self.ffmpeg_path,
            "nvidia_smi_path": self.nvidia_smi_path,
            "torch_error": self.torch_error,
            "onnxruntime_error": self.onnxruntime_error,
            "providers": list(self.providers),
        }


def _nvidia_smi_path() -> str:
    return shutil.which("nvidia-smi") or ""


def _nvidia_smi_works(path: str) -> bool:
    if not path:
        return False
    try:
        result = subprocess.run(
            [path, "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool((result.stdout or "").strip())


def guidance_for_report(report: AccelerationReport) -> str:
    """Return user-facing setup guidance for the current report."""
    guidance = {
        "ok": (
            "GPU acceleration is ready.\n"
            "The app will use GPU mode automatically for supported model features."
        ),
        "ffmpeg_missing": (
            "FFmpeg was not found.\n"
            "Set an FFmpeg override in Settings, or place FFmpeg on PATH."
        ),
        "no_nvidia_gpu": (
            "No usable NVIDIA GPU or driver was detected.\n"
            "The app can still run in CPU mode. Install an NVIDIA driver and CUDA-capable GPU"
            " if you want hardware acceleration."
        ),
        "driver_or_runtime_missing": (
            "An NVIDIA GPU appears to be present, but CUDA runtime support is incomplete.\n"
            "Update the NVIDIA driver first. If PyTorch and ONNX Runtime are still unavailable"
            " afterward, reinstall the CUDA-enabled Python packages used by this build."
        ),
        "torch_cuda_unavailable": (
            "PyTorch is installed, but CUDA is not usable from it.\n"
            "Check that the NVIDIA driver is current and that the packaged/runtime DLL paths"
            " are valid. If you use custom paths, verify them in Settings."
        ),
        "onnxruntime_cuda_unavailable": (
            "PyTorch can see the GPU, but ONNX Runtime does not expose CUDAExecutionProvider.\n"
            " Reinstall a CUDA-enabled ONNX Runtime build or correct the runtime DLL paths."
        ),
        "path_misconfiguration": (
            "One or more configured runtime paths are missing.\n"
            "Review custom runtime paths and the FFmpeg override in Settings."
        ),
        "pytorch_unavailable": (
            "PyTorch could not be imported.\n"
            "If you are using a packaged build, rebuild the runtime bundle. In source mode,"
            " reinstall the CUDA-enabled torch packages."
        ),
    }
    return guidance.get(report.reason_code, report.details)


def probe_acceleration(settings: "AppSettings | None" = None) -> AccelerationReport:
    """Inspect GPU/runtime readiness for the current process."""
    if settings is None:
        from .settings import get_app_settings

        settings = get_app_settings()
    settings.apply_runtime_environment()

    ffmpeg_path = get_ffmpeg_executable() or ""
    nvidia_smi = _nvidia_smi_path()
    configured_runtime_paths = settings.extra_runtime_paths()
    missing_runtime_paths = [path for path in configured_runtime_paths if not Path(path).exists()]

    if missing_runtime_paths:
        details = "Missing runtime path(s):\n" + "\n".join(missing_runtime_paths)
        return AccelerationReport(
            mode="cpu",
            reason_code="path_misconfiguration",
            summary="Custom runtime path configuration is incomplete.",
            details=details,
            ffmpeg_path=ffmpeg_path,
            nvidia_smi_path=nvidia_smi,
        )

    if not ffmpeg_path:
        return AccelerationReport(
            mode="cpu",
            reason_code="ffmpeg_missing",
            summary="FFmpeg was not detected.",
            details=guidance_for_report(
                AccelerationReport("cpu", "ffmpeg_missing", "", "", ffmpeg_path="", nvidia_smi_path=nvidia_smi)
            ),
            ffmpeg_path="",
            nvidia_smi_path=nvidia_smi,
        )

    try:
        import torch
    except Exception as exc:
        reason = "driver_or_runtime_missing" if _nvidia_smi_works(nvidia_smi) else "pytorch_unavailable"
        report = AccelerationReport(
            mode="cpu",
            reason_code=reason,
            summary="PyTorch could not be imported.",
            details="PyTorch import failed.\n\n" + str(exc),
            ffmpeg_path=ffmpeg_path,
            nvidia_smi_path=nvidia_smi,
            torch_error=str(exc),
        )
        report.details = guidance_for_report(report) + "\n\nOriginal error:\n" + str(exc)
        return report

    torch_version = getattr(torch, "__version__", "")
    cuda_version = str(getattr(getattr(torch, "version", object()), "cuda", "") or "")

    if not torch.cuda.is_available():
        reason = "torch_cuda_unavailable" if _nvidia_smi_works(nvidia_smi) else "no_nvidia_gpu"
        report = AccelerationReport(
            mode="cpu",
            reason_code=reason,
            summary="GPU acceleration is not available in PyTorch.",
            details="PyTorch is installed, but torch.cuda.is_available() returned False.",
            ffmpeg_path=ffmpeg_path,
            nvidia_smi_path=nvidia_smi,
            torch_version=torch_version,
            cuda_version=cuda_version,
        )
        report.details = guidance_for_report(report)
        return report

    gpu_name = ""
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "Detected GPU"

    try:
        import onnxruntime as ort

        providers = tuple(ort.get_available_providers())
        onnxruntime_version = getattr(ort, "__version__", "")
    except Exception as exc:
        report = AccelerationReport(
            mode="cpu",
            reason_code="driver_or_runtime_missing",
            summary="ONNX Runtime could not be imported.",
            details="ONNX Runtime import failed.\n\n" + str(exc),
            gpu_name=gpu_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            ffmpeg_path=ffmpeg_path,
            nvidia_smi_path=nvidia_smi,
            onnxruntime_error=str(exc),
        )
        report.details = guidance_for_report(report) + "\n\nOriginal error:\n" + str(exc)
        return report

    if "CUDAExecutionProvider" not in providers:
        report = AccelerationReport(
            mode="cpu",
            reason_code="onnxruntime_cuda_unavailable",
            summary="ONNX Runtime CUDA provider is unavailable.",
            details="Available ONNX Runtime providers: " + ", ".join(providers),
            gpu_name=gpu_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            onnxruntime_version=onnxruntime_version,
            ffmpeg_path=ffmpeg_path,
            nvidia_smi_path=nvidia_smi,
            providers=providers,
        )
        report.details = guidance_for_report(report) + "\n\nAvailable providers:\n" + ", ".join(providers)
        return report

    report = AccelerationReport(
        mode="gpu",
        reason_code="ok",
        summary=f"GPU ready: {gpu_name}",
        details="GPU acceleration is available for torch and ONNX Runtime.",
        gpu_name=gpu_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        onnxruntime_version=onnxruntime_version,
        ffmpeg_path=ffmpeg_path,
        nvidia_smi_path=nvidia_smi,
        providers=providers,
    )
    report.details = guidance_for_report(report)
    return report
