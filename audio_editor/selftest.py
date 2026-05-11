"""Headless validation helpers for source and frozen builds."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from .diagnostics import probe_acceleration
from .media_utils import extract_audio_from_video
from .runtime import (
    ensure_ffmpeg_environment,
    ensure_windows_app_id,
    get_icon_path,
    is_frozen,
)


def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _append_trace(path: str | Path, message: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _apply_selftest_runtime():
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    from .settings import get_app_settings

    settings = get_app_settings()
    settings.apply_runtime_environment()
    ensure_ffmpeg_environment()
    ensure_windows_app_id()
    return settings


def create_fixture_audio(output_path: str | Path, seconds: float = 2.5, sample_rate: int = 44100) -> Path:
    """Create a small stereo audio fixture."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    left = (
        0.42 * np.sin(2 * np.pi * 220 * t)
        + 0.18 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * np.sin(2 * np.pi * 880 * t)
    )
    right = (
        0.38 * np.sin(2 * np.pi * 247 * t)
        + 0.16 * np.sin(2 * np.pi * 494 * t)
        + 0.05 * np.sin(2 * np.pi * 988 * t)
    )
    sweep = 0.06 * np.sin(2 * np.pi * (90 + 180 * t / max(seconds, 0.1)) * t)
    audio = np.stack([left + sweep, right - sweep], axis=1)
    sf.write(output, np.clip(audio, -1.0, 1.0), sample_rate)
    return output


def create_fixture_video(audio_path: str | Path, output_path: str | Path) -> Path:
    """Create a tiny MP4 that carries the fixture audio."""
    import subprocess

    from .runtime import ffmpeg_command

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ffmpeg_command(
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=#162B45:s=1280x720:d=2.5",
            "-i",
            str(audio_path),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output),
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return output


def run_startup_smoke() -> dict:
    """Instantiate the main window offscreen and verify the icon is loaded."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _apply_selftest_runtime()
    from .main import create_application
    from .main_window import MainWindow

    app = create_application([])
    window = MainWindow()
    app.processEvents()
    icon = window.windowIcon()
    window.show()
    app.processEvents()

    window._main_tabs.setCurrentIndex(1)
    app.processEvents()
    enhance_switches: list[str] = []
    for idx in range(window._enhance_engine_combo.count()):
        window._enhance_engine_combo.setCurrentIndex(idx)
        app.processEvents()
        enhance_switches.append(window._enhance_engine_combo.itemText(idx))

    window._main_tabs.setCurrentIndex(0)
    app.processEvents()
    separation_categories = []
    for idx in range(window._sub_cat_combo.count()):
        window._sub_cat_combo.setCurrentIndex(idx)
        app.processEvents()
        separation_categories.append(window._sub_cat_combo.itemText(idx))

    result = {
        "window_title": window.windowTitle(),
        "icon_loaded": not icon.isNull(),
        "icon_path": str(get_icon_path()) if get_icon_path() else None,
        "window_size": [window.width(), window.height()],
        "minimum_size": [window.minimumWidth(), window.minimumHeight()],
        "enhance_engines": [
            window._enhance_engine_combo.itemText(i)
            for i in range(window._enhance_engine_combo.count())
        ],
        "left_scroll_widget_resizable": (
            window._left_scroll.widgetResizable() if window._left_scroll is not None else None
        ),
        "left_scroll_vpolicy": (
            window._left_scroll.verticalScrollBarPolicy().name
            if window._left_scroll is not None
            else None
        ),
        "clearvoice_models": [
            str(window._cv_model_combo.itemData(i))
            for i in range(window._cv_model_combo.count())
        ],
        "mdxnet_models": [
            str(window._mdx_model_combo.itemData(i))
            for i in range(window._mdx_model_combo.count())
        ],
        "voicefixer_modes": [
            window._vf_mode_combo.itemText(i)
            for i in range(window._vf_mode_combo.count())
        ],
        "enhance_switches": enhance_switches,
        "separation_categories": separation_categories,
    }
    window.close()
    app.processEvents()
    return result


def run_backend_availability() -> dict:
    """Check optional backend imports without running their full UI flows."""
    _apply_selftest_runtime()
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    from .enhance_backend import (
        check_audio_separator_available,
        check_clearvoice_available,
        check_deepfilter_available,
        check_enhance_available,
        check_metricgan_available,
        check_voicefixer_available,
    )
    from .sam_backend import check_sam_available

    return {
        "sam_audio_available": check_sam_available(),
        "resemble_enhance_available": check_enhance_available(),
        "deepfilternet_available": check_deepfilter_available(),
        "clearvoice_available": check_clearvoice_available(),
        "audio_separator_available": check_audio_separator_available(),
        "voicefixer_available": check_voicefixer_available(),
        "metricgan_available": check_metricgan_available(),
        "frozen": is_frozen(),
    }


def run_individual_availability(kind: str) -> dict:
    """Check one optional backend import in isolation."""
    _apply_selftest_runtime()
    if kind == "sam":
        from .sam_backend import check_sam_available

        return {"sam_audio_available": check_sam_available(), "frozen": is_frozen()}
    if kind == "enhance":
        from .enhance_backend import check_enhance_available

        return {"resemble_enhance_available": check_enhance_available(), "frozen": is_frozen()}
    if kind == "deepfilter":
        from .enhance_backend import check_deepfilter_available

        return {"deepfilternet_available": check_deepfilter_available(), "frozen": is_frozen()}
    if kind == "clearvoice":
        from .enhance_backend import check_clearvoice_available

        return {"clearvoice_available": check_clearvoice_available(), "frozen": is_frozen()}
    if kind == "audio-separator":
        from .enhance_backend import check_audio_separator_available

        return {"audio_separator_available": check_audio_separator_available(), "frozen": is_frozen()}
    if kind == "voicefixer":
        from .enhance_backend import check_voicefixer_available

        return {"voicefixer_available": check_voicefixer_available(), "frozen": is_frozen()}
    if kind == "metricgan":
        from .enhance_backend import check_metricgan_available

        return {"metricgan_available": check_metricgan_available(), "frozen": is_frozen()}
    raise ValueError(f"Unsupported availability kind: {kind}")


def run_enhancement_regressions(workdir: str | Path) -> dict:
    """Validate enhancement model metadata and local preprocessing helpers."""
    import torch

    from .audio_engine import AudioEngine
    from .enhance_backend import (
        CLEARVOICE_MODEL_SAMPLE_RATES,
        MDXNET_ENHANCEMENT_PRESETS,
        VOICEFIXER_SUPPORTED_MODES,
        _enhance_metricgan_in_chunks,
        _prepare_audio_for_model,
    )
    from .separator_backend import get_preset_by_name

    root = _ensure_dir(workdir)
    fixtures_dir = _ensure_dir(root / "fixtures")
    prepared_dir = _ensure_dir(root / "prepared")
    fixture = create_fixture_audio(fixtures_dir / "enhance_input.wav", seconds=3.0, sample_rate=44100)

    prepared_reports = {}
    for model_name, target_sr in CLEARVOICE_MODEL_SAMPLE_RATES.items():
        prepared_path, metadata = _prepare_audio_for_model(
            str(fixture),
            prepared_dir / model_name,
            target_sample_rate=target_sr,
            mono=True,
            stem_suffix=model_name.lower(),
        )
        info = sf.info(prepared_path)
        prepared_reports[model_name] = {
            "prepared_path": prepared_path,
            "target_sample_rate": target_sr,
            "prepared_sample_rate": info.samplerate,
            "prepared_channels": info.channels,
            "original_sample_rate": metadata["original_sample_rate"],
            "prepared_ok": info.samplerate == target_sr and info.channels == 1,
        }

    class _IdentityMetricGan:
        def enhance_batch(self, audio, lengths):
            return audio

    metricgan_input = torch.linspace(-0.35, 0.35, steps=16_000 * 50, dtype=torch.float32).unsqueeze(0)
    metricgan_output = _enhance_metricgan_in_chunks(_IdentityMetricGan(), metricgan_input)
    metricgan_max_abs_error = float(torch.max(torch.abs(metricgan_output - metricgan_input)).item())

    mdx_reports = {}
    for preset_name in MDXNET_ENHANCEMENT_PRESETS:
        preset = get_preset_by_name(preset_name)
        mdx_reports[preset_name] = {
            "exists": preset is not None,
            "architecture": preset.architecture if preset is not None else None,
            "stems": list(preset.stems) if preset is not None else [],
            "valid": preset is not None and preset.architecture == "MDX-Net" and "vocals" in preset.stems,
        }

    mdx_output_labels = {
        "speaker": AudioEngine._display_name_for_stem("demo_mdxnet_speaker_voice.wav"),
        "background": AudioEngine._display_name_for_stem("demo_mdxnet_background_bleed_room.wav"),
    }

    return {
        "clearvoice_model_sample_rates": CLEARVOICE_MODEL_SAMPLE_RATES,
        "prepared_models": prepared_reports,
        "mdxnet_enhancement_models": mdx_reports,
        "mdxnet_output_labels": mdx_output_labels,
        "voicefixer_supported_modes": list(VOICEFIXER_SUPPORTED_MODES),
        "metricgan_identity_shape": list(metricgan_output.shape),
        "metricgan_identity_max_abs_error": metricgan_max_abs_error,
        "metricgan_identity_ok": metricgan_max_abs_error < 1e-5,
    }


def run_video_extract_smoke(video_path: str | Path, output_dir: str | Path) -> dict:
    """Verify the video extraction feature produces a readable WAV."""
    output = extract_audio_from_video(str(video_path), str(output_dir))
    info = sf.info(output)
    return {
        "output_path": output,
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }


def run_separator_import_chain(workdir: str | Path) -> dict:
    """Import separator dependencies one by one and record progress."""
    trace_path = Path(workdir) / "separator_import_trace.log"
    modules = [
        "librosa",
        "numpy",
        "yaml",
        "requests",
        "torch",
        "onnxruntime",
        "tqdm",
        "audio_separator.separator.ensembler",
    ]
    imported: list[str] = []
    for name in modules:
        _append_trace(trace_path, f"import-chain: importing {name}")
        __import__(name)
        imported.append(name)
        _append_trace(trace_path, f"import-chain: imported {name}")
    return {"imported": imported}


def run_separator_smoke(
    audio_path: str | Path,
    output_dir: str | Path,
    model_filename: str,
) -> dict:
    """Run one real audio-separator inference."""
    trace_path = Path(output_dir) / "separator_trace.log"
    _append_trace(trace_path, "separator smoke: start")
    _append_trace(trace_path, "separator smoke: importing audio_separator.separator")
    from audio_separator.separator import Separator
    _append_trace(trace_path, "separator smoke: imported audio_separator.separator")

    settings = _apply_selftest_runtime()
    model_dir = Path(settings.separator_model_dir())
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_filename
    existed_before = model_path.exists()

    _append_trace(trace_path, "separator smoke: creating Separator")
    separator = Separator(
        output_dir=str(output_dir),
        output_format="wav",
        model_file_dir=str(model_dir),
    )
    _append_trace(trace_path, "separator smoke: loading model")
    separator.load_model(model_filename=model_filename)
    _append_trace(trace_path, "separator smoke: separating audio")
    outputs = separator.separate(str(audio_path))
    _append_trace(trace_path, "separator smoke: separation finished")

    resolved_outputs = []
    for item in outputs:
        path = Path(item)
        if not path.is_absolute():
            path = Path(output_dir) / path
        info = sf.info(path)
        resolved_outputs.append({
            "path": str(path.resolve()),
            "frames": info.frames,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        })

    _append_trace(trace_path, "separator smoke: outputs verified")
    return {
        "model_filename": model_filename,
        "model_existed_before": existed_before,
        "model_exists_after": model_path.exists(),
        "output_count": len(resolved_outputs),
        "outputs": resolved_outputs,
    }


def run_suite(workdir: str | Path, model_filename: str) -> dict:
    """Run the full validation suite."""
    _apply_selftest_runtime()
    root = _ensure_dir(workdir)
    fixtures_dir = _ensure_dir(root / "fixtures")
    outputs_dir = _ensure_dir(root / "outputs")

    audio_fixture = create_fixture_audio(fixtures_dir / "smoke_input.wav")
    video_fixture = create_fixture_video(audio_fixture, fixtures_dir / "smoke_input.mp4")

    report = {
        "startup": run_startup_smoke(),
        "availability": run_backend_availability(),
        "enhancement_regressions": run_enhancement_regressions(outputs_dir / "enhancement_regressions"),
        "video_extract": run_video_extract_smoke(video_fixture, outputs_dir / "video_extract"),
        "separator": run_separator_smoke(audio_fixture, outputs_dir / "separator", model_filename),
    }
    return report


def run_gpu_stack() -> dict:
    """Report current GPU/runtime readiness."""
    settings = _apply_selftest_runtime()
    return probe_acceleration(settings).to_dict()


def run_settings_roundtrip(workdir: str | Path) -> dict:
    """Verify settings persistence with an isolated settings file."""
    from .settings import ENV_SETTINGS_PATH, get_app_settings

    root = _ensure_dir(workdir)
    settings_path = root / "settings_roundtrip.ini"
    runtime_a = _ensure_dir(root / "runtimeA")
    runtime_b = _ensure_dir(root / "runtimeB")
    output_dir = _ensure_dir(root / "exports")
    separator_dir = _ensure_dir(root / "models" / "separator")
    hf_cache_dir = _ensure_dir(root / "cache" / "huggingface")
    deepfilter_dir = _ensure_dir(root / "models" / "DeepFilterNet")

    previous = os.environ.get(ENV_SETTINGS_PATH)
    try:
        os.environ[ENV_SETTINGS_PATH] = str(settings_path)
        settings = get_app_settings(reset=True)
        settings.set_output_mode("fixed")
        settings.set_default_output_dir(str(output_dir))
        settings.set_reuse_last_used_folder(False)
        settings.set_separator_model_dir(str(separator_dir))
        settings.set_hf_cache_dir(str(hf_cache_dir))
        settings.set_deepfilter_model_dir(str(deepfilter_dir))
        settings.set_ffmpeg_override("")
        settings.set_extra_runtime_paths([str(runtime_a), str(runtime_b)])
        settings.sync()

        reloaded = get_app_settings(reset=True)
        report = {
            "settings_path": str(reloaded.path),
            "output_mode": reloaded.output_mode(),
            "default_output_dir": reloaded.default_output_dir(),
            "reuse_last_used_folder": reloaded.reuse_last_used_folder(),
            "separator_model_dir": reloaded.separator_model_dir(),
            "hf_cache_dir": reloaded.hf_cache_dir(),
            "deepfilter_model_dir": reloaded.deepfilter_model_dir(),
            "extra_runtime_paths": reloaded.extra_runtime_paths(),
        }
        report["roundtrip_ok"] = (
            report["output_mode"] == "fixed"
            and Path(report["default_output_dir"]).resolve() == output_dir.resolve()
            and report["reuse_last_used_folder"] is False
            and Path(report["separator_model_dir"]).resolve() == separator_dir.resolve()
            and Path(report["hf_cache_dir"]).resolve() == hf_cache_dir.resolve()
            and Path(report["deepfilter_model_dir"]).resolve() == deepfilter_dir.resolve()
            and report["extra_runtime_paths"] == [str(runtime_a.resolve()), str(runtime_b.resolve())]
        )
        return report
    finally:
        if previous is None:
            os.environ.pop(ENV_SETTINGS_PATH, None)
        else:
            os.environ[ENV_SETTINGS_PATH] = previous
        get_app_settings(reset=True)


def run_settings_model_scan(workdir: str | Path) -> dict:
    """Create a synthetic model tree, scan it, then verify deletion helpers."""
    from .settings import ENV_SETTINGS_PATH, get_app_settings

    root = _ensure_dir(workdir)
    settings_path = root / "settings_model_scan.ini"
    separator_dir = _ensure_dir(root / "models" / "separator")
    hf_cache_dir = _ensure_dir(root / "cache" / "huggingface")
    deepfilter_dir = _ensure_dir(root / "models" / "DeepFilterNet")

    (separator_dir / "Kim_Vocal_2.onnx").write_bytes(b"separator-model")
    (separator_dir / "config.yaml").write_text("model: separator\n", encoding="utf-8")

    hf_repo = _ensure_dir(hf_cache_dir / "models--example--sam-audio-large")
    _ensure_dir(hf_repo / "refs")
    _ensure_dir(hf_repo / "snapshots" / "123abc")
    (hf_repo / "snapshots" / "123abc" / "checkpoint.pt").write_bytes(b"hf-model")

    deepfilter_model = _ensure_dir(deepfilter_dir / "DeepFilterNet3")
    (deepfilter_model / "config.ini").write_text("[model]\nname=DeepFilterNet3\n", encoding="utf-8")
    _ensure_dir(deepfilter_model / "checkpoints")
    (deepfilter_model / "checkpoints" / "best.ckpt").write_bytes(b"deepfilter-model")

    previous = os.environ.get(ENV_SETTINGS_PATH)
    try:
        os.environ[ENV_SETTINGS_PATH] = str(settings_path)
        settings = get_app_settings(reset=True)
        settings.set_separator_model_dir(str(separator_dir))
        settings.set_hf_cache_dir(str(hf_cache_dir))
        settings.set_deepfilter_model_dir(str(deepfilter_dir))
        settings.sync()

        entries = settings.model_entries()
        before = [
            {
                "backend": entry.backend,
                "name": entry.name,
                "path": str(entry.path),
                "size_bytes": entry.size_bytes,
            }
            for entry in entries
        ]
        settings.clear_all_downloaded_models()
        after = [
            {
                "backend": entry.backend,
                "name": entry.name,
                "path": str(entry.path),
                "size_bytes": entry.size_bytes,
            }
            for entry in settings.model_entries()
        ]
        return {
            "settings_path": str(settings.path),
            "entries_before": before,
            "entries_after": after,
            "scan_ok": len(before) >= 3 and len(after) == 0,
        }
    finally:
        if previous is None:
            os.environ.pop(ENV_SETTINGS_PATH, None)
        else:
            os.environ[ENV_SETTINGS_PATH] = previous
        get_app_settings(reset=True)


def _write_report(path: str | None, report: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ai-audio-toolkit self-tests.")
    parser.add_argument(
        "--mode",
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
    parser.add_argument("--workdir", required=True, help="Directory for generated fixtures and outputs.")
    parser.add_argument("--report", help="Optional JSON report path.")
    parser.add_argument(
        "--model-filename",
        default="kuielab_a_drums.onnx",
        help="audio-separator model filename to validate.",
    )
    args = parser.parse_args(argv)

    try:
        if args.mode == "startup":
            report = {"startup": run_startup_smoke()}
        elif args.mode == "availability":
            report = {"availability": run_backend_availability()}
        elif args.mode == "sam":
            report = {"availability": run_individual_availability("sam")}
        elif args.mode == "enhance":
            report = {"availability": run_individual_availability("enhance")}
        elif args.mode == "deepfilter":
            report = {"availability": run_individual_availability("deepfilter")}
        elif args.mode == "clearvoice":
            report = {"availability": run_individual_availability("clearvoice")}
        elif args.mode == "audio-separator":
            report = {"availability": run_individual_availability("audio-separator")}
        elif args.mode == "voicefixer":
            report = {"availability": run_individual_availability("voicefixer")}
        elif args.mode == "metricgan":
            report = {"availability": run_individual_availability("metricgan")}
        elif args.mode == "enhancement-regressions":
            report = {"enhancement_regressions": run_enhancement_regressions(args.workdir)}
        elif args.mode == "gpu-stack":
            report = {"gpu_stack": run_gpu_stack()}
        elif args.mode == "settings-roundtrip":
            report = {"settings_roundtrip": run_settings_roundtrip(args.workdir)}
        elif args.mode == "settings-model-scan":
            report = {"settings_model_scan": run_settings_model_scan(args.workdir)}
        elif args.mode == "separator-imports":
            report = {"separator_imports": run_separator_import_chain(args.workdir)}
        elif args.mode == "video":
            root = _ensure_dir(args.workdir)
            fixtures_dir = _ensure_dir(root / "fixtures")
            outputs_dir = _ensure_dir(root / "outputs")
            audio_fixture = create_fixture_audio(fixtures_dir / "smoke_input.wav")
            video_fixture = create_fixture_video(audio_fixture, fixtures_dir / "smoke_input.mp4")
            report = {"video_extract": run_video_extract_smoke(video_fixture, outputs_dir / "video_extract")}
        elif args.mode == "separator":
            root = _ensure_dir(args.workdir)
            fixtures_dir = _ensure_dir(root / "fixtures")
            outputs_dir = _ensure_dir(root / "outputs")
            audio_fixture = create_fixture_audio(fixtures_dir / "smoke_input.wav")
            report = {
                "separator": run_separator_smoke(
                    audio_fixture,
                    outputs_dir / "separator",
                    args.model_filename,
                )
            }
        else:
            report = run_suite(args.workdir, args.model_filename)

        report["status"] = "ok"
        _write_report(args.report, report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_report(args.report, report)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
