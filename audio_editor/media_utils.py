"""Helpers for working with audio/video media files."""
from __future__ import annotations

import subprocess
from pathlib import Path

import soundfile as sf

from .runtime import ffmpeg_command, get_ffprobe_executable


def extract_audio_from_video_to_path(
    video_path: str,
    output_path: str,
    sample_rate: int = 44100,
    channels: int = 2,
) -> str:
    """Extract a WAV file from a video to an explicit output path."""
    video = Path(video_path)
    wav_path = Path(output_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ffmpeg_command(
                "-y",
                "-i",
                str(video),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(wav_path),
            ),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or "FFmpeg failed to extract audio from the video.") from exc

    return str(wav_path)


def extract_audio_from_video(
    video_path: str,
    output_dir: str,
    sample_rate: int = 44100,
    channels: int = 2,
) -> str:
    """Extract a WAV file from a video and return the output path."""
    video = Path(video_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    wav_path = target_dir / f"{video.stem}_extracted.wav"
    return extract_audio_from_video_to_path(
        video_path,
        str(wav_path),
        sample_rate=sample_rate,
        channels=channels,
    )


def probe_media_duration(path: str) -> float:
    """Return the media duration in seconds when it can be determined."""
    ffprobe = get_ffprobe_executable()
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            value = (result.stdout or "").strip()
            if value:
                return max(0.0, float(value))
        except (subprocess.CalledProcessError, ValueError, OSError):
            pass

    try:
        info = sf.info(path)
    except RuntimeError:
        return 0.0
    return float(info.duration or 0.0)


def export_media_clip(
    source_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    *,
    audio_only: bool,
) -> str:
    """Export a clipped section of audio or video via FFmpeg."""
    start = max(0.0, float(start_time))
    end = max(start, float(end_time))
    duration = max(0.01, end - start)

    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    command = [
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
    ]

    if audio_only:
        command.extend(["-vn"])
        if suffix == ".mp3":
            command.extend(["-c:a", "libmp3lame", "-q:a", "2"])
        elif suffix in {".m4a", ".aac"}:
            command.extend(["-c:a", "aac", "-b:a", "192k"])
        elif suffix == ".flac":
            command.extend(["-c:a", "flac"])
        else:
            command.extend(["-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2"])
    else:
        command.extend(
            [
                "-map",
                "0:v:0?",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
            ]
        )

    command.append(str(output))

    try:
        subprocess.run(
            ffmpeg_command(*command),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or "FFmpeg failed to export the selected clip.") from exc

    return str(output)
