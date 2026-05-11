"""Persistent application settings and model storage helpers."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QSettings

from .branding import APP_DISPLAY_NAME, LEGACY_APP_NAME, LEGACY_DOT_DIR

APP_NAME = APP_DISPLAY_NAME
ENV_SETTINGS_PATH = "AI_AUDIO_TOOLKIT_SETTINGS_PATH"
LEGACY_ENV_SETTINGS_PATH = "AUDIO_EDITOR_SETTINGS_PATH"
ENV_BUNDLE_ROOT = "AI_AUDIO_TOOLKIT_BUNDLE_ROOT"
LEGACY_ENV_BUNDLE_ROOT = "AUDIO_EDITOR_BUNDLE_ROOT"
_SETTINGS_SINGLETON: "AppSettings | None" = None


@dataclass(slots=True)
class ModelEntry:
    """A discovered downloadable model or cache entry."""

    backend: str
    name: str
    path: Path
    size_bytes: int


def _local_app_data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def _legacy_local_app_data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / LEGACY_APP_NAME
    return Path.home() / ".local" / "share" / LEGACY_APP_NAME


def legacy_data_root() -> Path:
    """Return the legacy per-user data root used by earlier builds."""
    return Path.home() / LEGACY_DOT_DIR


def legacy_data_roots() -> list[Path]:
    """Return old data roots that may contain user settings or models."""
    return [_legacy_local_app_data_root(), legacy_data_root()]


def default_data_root() -> Path:
    """Return the default application data root."""
    return _local_app_data_root()


def default_separator_model_dir() -> Path:
    return default_data_root() / "models" / "audio_separator"


def default_hf_cache_dir() -> Path:
    return default_data_root() / "cache" / "huggingface"


def default_deepfilter_model_dir() -> Path:
    return default_data_root() / "models" / "DeepFilterNet"


def settings_file_path() -> Path:
    """Return the concrete settings file path."""
    override = os.environ.get(ENV_SETTINGS_PATH) or os.environ.get(LEGACY_ENV_SETTINGS_PATH)
    if override:
        return Path(override).expanduser().resolve()
    return default_data_root() / "settings.ini"


def _legacy_settings_file_candidates() -> list[Path]:
    return [root / "settings.ini" for root in legacy_data_roots()]


def _normalize_path(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _split_multiline(value: str | None) -> list[str]:
    if not value:
        return []
    items: list[str] = []
    for line in value.replace(";", "\n").splitlines():
        cleaned = _normalize_path(line)
        if cleaned and cleaned not in items:
            items.append(cleaned)
    return items


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _discover_separator_models(root: Path) -> list[ModelEntry]:
    if not root.exists():
        return []

    result: list[ModelEntry] = []
    seen: set[Path] = set()
    patterns = ("*.onnx", "*.ckpt", "*.pth", "*.yaml")
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            result.append(
                ModelEntry(
                    backend="Separator",
                    name=path.name,
                    path=path,
                    size_bytes=path.stat().st_size,
                )
            )
    return sorted(result, key=lambda item: (item.backend, item.name.lower()))


def _discover_hf_models(root: Path) -> list[ModelEntry]:
    if not root.exists():
        return []

    result: list[ModelEntry] = []
    for path in sorted(root.iterdir(), key=lambda entry: entry.name.lower()):
        if not path.exists():
            continue
        if path.is_dir() and (
            path.name.startswith("models--")
            or path.name.startswith("datasets--")
            or {"refs", "snapshots"}.issubset({child.name for child in path.iterdir() if child.is_dir()})
        ):
            result.append(
                ModelEntry(
                    backend="HuggingFace",
                    name=path.name.replace("models--", "").replace("--", "/"),
                    path=path,
                    size_bytes=_directory_size(path),
                )
            )
    return result


def _discover_deepfilter_models(root: Path) -> list[ModelEntry]:
    if not root.exists():
        return []

    result: list[ModelEntry] = []
    for path in sorted(root.iterdir(), key=lambda entry: entry.name.lower()):
        if not path.is_dir():
            continue
        has_model_layout = (path / "config.ini").is_file() or (path / "checkpoints").is_dir()
        if not has_model_layout:
            continue
        result.append(
            ModelEntry(
                backend="DeepFilterNet",
                name=path.name,
                path=path,
                size_bytes=_directory_size(path),
            )
        )
    return result


def delete_model_entry(entry: ModelEntry) -> None:
    """Delete a discovered model entry from disk."""
    if entry.path.is_dir():
        shutil.rmtree(entry.path)
    elif entry.path.exists():
        entry.path.unlink()


class AppSettings:
    """Typed settings wrapper for the application."""

    def __init__(self) -> None:
        path = settings_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            for legacy_path in _legacy_settings_file_candidates():
                if legacy_path.is_file():
                    try:
                        shutil.copy2(legacy_path, path)
                    except OSError:
                        pass
                    break
        self.path = path
        self._settings = QSettings(str(path), QSettings.Format.IniFormat)
        self._bootstrap_defaults()

    def _bootstrap_defaults(self) -> None:
        if self._settings.value("meta/version") is not None:
            return

        self._settings.setValue("meta/version", 1)
        self._settings.setValue("general/output_mode", "per_input")
        self._settings.setValue("general/default_output_dir", "")
        self._settings.setValue("general/reuse_last_used_folder", True)
        self._settings.setValue("general/last_input_dir", "")
        self._settings.setValue("general/last_output_dir", "")

        separator_dir = default_separator_model_dir()
        ffmpeg_override = ""
        for legacy_root in legacy_data_roots():
            if not legacy_root.exists():
                continue
            legacy_models = legacy_root / "models"
            if legacy_models.exists():
                separator_dir = legacy_models
            legacy_ffmpeg = (
                legacy_root
                / "ffmpeg-shared"
                / "ffmpeg-8.1-full_build-shared"
                / "bin"
                / "ffmpeg.exe"
            )
            if legacy_ffmpeg.is_file():
                ffmpeg_override = str(legacy_ffmpeg.resolve())
                break

        self._settings.setValue("paths/separator_model_dir", str(separator_dir))
        self._settings.setValue("paths/hf_cache_dir", str(default_hf_cache_dir()))
        self._settings.setValue("paths/deepfilter_model_dir", str(default_deepfilter_model_dir()))
        self._settings.setValue("paths/ffmpeg_override", ffmpeg_override)
        self._settings.setValue("paths/extra_runtime_paths", "")
        self._settings.sync()

    def sync(self) -> None:
        self._settings.sync()

    def output_mode(self) -> str:
        return str(self._settings.value("general/output_mode", "per_input"))

    def set_output_mode(self, mode: str) -> None:
        self._settings.setValue("general/output_mode", mode)

    def default_output_dir(self) -> str:
        return _normalize_path(self._settings.value("general/default_output_dir", ""))

    def set_default_output_dir(self, path: str) -> None:
        self._settings.setValue("general/default_output_dir", _normalize_path(path))

    def reuse_last_used_folder(self) -> bool:
        return _to_bool(self._settings.value("general/reuse_last_used_folder", True), True)

    def set_reuse_last_used_folder(self, value: bool) -> None:
        self._settings.setValue("general/reuse_last_used_folder", bool(value))

    def last_input_dir(self) -> str:
        return _normalize_path(self._settings.value("general/last_input_dir", ""))

    def set_last_input_dir(self, path: str) -> None:
        self._settings.setValue("general/last_input_dir", _normalize_path(path))

    def last_output_dir(self) -> str:
        return _normalize_path(self._settings.value("general/last_output_dir", ""))

    def set_last_output_dir(self, path: str) -> None:
        self._settings.setValue("general/last_output_dir", _normalize_path(path))

    def separator_model_dir(self) -> str:
        return _normalize_path(
            self._settings.value("paths/separator_model_dir", str(default_separator_model_dir()))
        )

    def set_separator_model_dir(self, path: str) -> None:
        self._settings.setValue("paths/separator_model_dir", _normalize_path(path))

    def hf_cache_dir(self) -> str:
        return _normalize_path(
            self._settings.value("paths/hf_cache_dir", str(default_hf_cache_dir()))
        )

    def set_hf_cache_dir(self, path: str) -> None:
        self._settings.setValue("paths/hf_cache_dir", _normalize_path(path))

    def deepfilter_model_dir(self) -> str:
        return _normalize_path(
            self._settings.value("paths/deepfilter_model_dir", str(default_deepfilter_model_dir()))
        )

    def set_deepfilter_model_dir(self, path: str) -> None:
        self._settings.setValue("paths/deepfilter_model_dir", _normalize_path(path))

    def ffmpeg_override(self) -> str:
        return _normalize_path(self._settings.value("paths/ffmpeg_override", ""))

    def set_ffmpeg_override(self, path: str) -> None:
        self._settings.setValue("paths/ffmpeg_override", _normalize_path(path))

    def extra_runtime_paths(self) -> list[str]:
        return _split_multiline(str(self._settings.value("paths/extra_runtime_paths", "")))

    def set_extra_runtime_paths(self, paths: list[str]) -> None:
        normalized = []
        for path in paths:
            cleaned = _normalize_path(path)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        self._settings.setValue("paths/extra_runtime_paths", "\n".join(normalized))

    def apply_runtime_environment(self) -> None:
        """Export app-managed runtime paths into process environment."""
        separator_dir = self.separator_model_dir()
        hf_cache = self.hf_cache_dir()
        deepfilter_dir = self.deepfilter_model_dir()

        if separator_dir:
            os.environ["AUDIO_SEPARATOR_MODEL_DIR"] = separator_dir
        if hf_cache:
            os.environ["HF_HOME"] = hf_cache
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
        if deepfilter_dir:
            os.environ["AI_AUDIO_TOOLKIT_DEEPFILTER_MODEL_DIR"] = deepfilter_dir
            os.environ["AUDIO_EDITOR_DEEPFILTER_MODEL_DIR"] = deepfilter_dir

    def model_entries(self) -> list[ModelEntry]:
        """Return discovered downloadable models across all backends."""
        entries = []
        entries.extend(_discover_separator_models(Path(self.separator_model_dir())))
        entries.extend(_discover_hf_models(Path(self.hf_cache_dir())))
        entries.extend(_discover_deepfilter_models(Path(self.deepfilter_model_dir())))
        return entries

    def clear_all_downloaded_models(self) -> None:
        for entry in self.model_entries():
            delete_model_entry(entry)


def get_app_settings(*, reset: bool = False) -> AppSettings:
    """Return a process-wide settings instance."""
    global _SETTINGS_SINGLETON
    if reset or _SETTINGS_SINGLETON is None:
        _SETTINGS_SINGLETON = AppSettings()
    return _SETTINGS_SINGLETON
