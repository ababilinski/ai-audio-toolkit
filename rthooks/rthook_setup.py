"""PyInstaller runtime hook — runs before any user imports.

Registers native DLL directories so PyTorch, ONNX Runtime, and DeepFilterNet
can load their CUDA/C-extension DLLs in the frozen one-dir bundle.
"""
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    # In PyInstaller one-dir mode the EXE and all collected files share one folder.
    base = Path(sys.executable).parent

    candidates = [
        base / "torch" / "lib",      # torch CUDA DLLs (collect_all layout)
        base / "torch.libs",          # alternate wheel layout
        base / "onnxruntime" / "capi",
        base / "DeepFilterLib.libs",
        base / "av.libs",
        base,                          # flat DLLs copied to root
    ]

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []

    for p in candidates:
        if not p.is_dir():
            continue
        p_str = str(p)
        try:
            os.add_dll_directory(p_str)
        except (AttributeError, OSError):
            pass
        if p_str not in path_parts:
            path_parts.insert(0, p_str)

    os.environ["PATH"] = os.pathsep.join(path_parts)
