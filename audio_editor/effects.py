"""Audio effects engine using numpy/scipy for live layered playback effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt


@dataclass
class EQBand:
    frequency: float  # Hz
    gain_db: float    # -12 to +12
    q: float = 1.0


@dataclass(frozen=True)
class EffectSpec:
    key: str
    label: str
    section: str
    description: str
    default_amount: float = 0.35
    min_amount: float = 0.0
    max_amount: float = 1.0


def apply_eq(audio: np.ndarray, sr: int, bands: list[EQBand]) -> np.ndarray:
    output = audio.copy().astype(np.float64)
    for band in bands:
        if abs(band.gain_db) < 0.1:
            continue
        w0 = 2 * np.pi * band.frequency / sr
        A = 10 ** (band.gain_db / 40)
        alpha = np.sin(w0) / (2 * band.q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

        if output.ndim == 2:
            for ch in range(output.shape[1]):
                output[:, ch] = sosfiltfilt(sos, output[:, ch])
        else:
            output = sosfiltfilt(sos, output)
    return output.astype(np.float32)


def apply_compressor(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
) -> np.ndarray:
    output = audio.copy().astype(np.float64)
    threshold = 10 ** (threshold_db / 20)
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))

    if output.ndim == 2:
        env_signal = np.max(np.abs(output), axis=1)
    else:
        env_signal = np.abs(output)

    envelope = np.zeros_like(env_signal)
    gain = np.ones_like(env_signal)

    for i in range(1, len(env_signal)):
        if env_signal[i] > envelope[i - 1]:
            envelope[i] = attack_coeff * envelope[i - 1] + (1 - attack_coeff) * env_signal[i]
        else:
            envelope[i] = release_coeff * envelope[i - 1] + (1 - release_coeff) * env_signal[i]

        if envelope[i] > threshold:
            gain_db_val = threshold_db + (20 * np.log10(envelope[i] + 1e-10) - threshold_db) / ratio
            gain[i] = 10 ** (gain_db_val / 20) / (envelope[i] + 1e-10)

    if output.ndim == 2:
        output *= gain[:, np.newaxis]
    else:
        output *= gain
    return output.astype(np.float32)


def apply_stereo_widening(audio: np.ndarray, width: float = 1.5) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio.astype(np.float32)
    mid = (audio[:, 0] + audio[:, 1]) / 2
    side = (audio[:, 0] - audio[:, 1]) / 2
    left = mid + side * width
    right = mid - side * width
    return np.column_stack([left, right]).astype(np.float32)


def normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return audio.astype(np.float32)
    target = 10 ** (target_db / 20)
    return (audio * (target / peak)).astype(np.float32)


def apply_limiter(audio: np.ndarray, ceiling_db: float = -0.3) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20)
    return np.clip(audio, -ceiling, ceiling).astype(np.float32)


def _ensure_2d(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def _blend(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    mix = float(np.clip(amount, 0.0, 1.0))
    if mix <= 0.0:
        return dry.astype(np.float32)
    return ((1.0 - mix) * dry + mix * wet).astype(np.float32)


def _signed_delta_blend(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    mix = float(np.clip(amount, -1.0, 1.0))
    if abs(mix) <= 1e-6:
        return dry.astype(np.float32)
    return np.clip(dry + mix * (wet - dry), -1.0, 1.0).astype(np.float32)


def _process_each_channel(audio: np.ndarray, fn) -> np.ndarray:
    dry = _ensure_2d(audio).astype(np.float64)
    wet = np.zeros_like(dry)
    for ch in range(dry.shape[1]):
        wet[:, ch] = fn(dry[:, ch])
    return wet.astype(np.float32)


def _simple_delay(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    delay_ms = 120.0 + 280.0 * amount
    delay_samples = max(1, int(sr * delay_ms / 1000.0))
    feedback = 0.18 + 0.35 * amount
    wet = dry.astype(np.float64).copy()
    for repeat, weight in enumerate([feedback, feedback * 0.6, feedback * 0.35], start=1):
        shift = delay_samples * repeat
        if shift >= len(dry):
            break
        wet[shift:] += dry[:-shift] * weight
    return _blend(dry, wet.astype(np.float32), amount * 0.75)


def _simple_reverb(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    ir_len = max(64, int(sr * (0.12 + 0.6 * amount)))
    t = np.linspace(0.0, 1.0, ir_len, endpoint=False)
    ir = np.exp(-t * (6.0 - 3.0 * amount))
    for tap_ms, gain in [(13, 0.45), (29, 0.32), (47, 0.22), (73, 0.14)]:
        idx = int(sr * tap_ms / 1000.0)
        if idx < ir_len:
            ir[idx] += gain * (0.6 + amount * 0.4)
    ir /= max(np.sum(np.abs(ir)), 1e-6)

    def _channel_reverb(channel: np.ndarray) -> np.ndarray:
        return fftconvolve(channel, ir, mode="full")[: len(channel)]

    wet = _process_each_channel(dry, _channel_reverb)
    shaped = dry * (1.0 - amount * 0.25) + wet * (0.55 + 0.25 * amount)
    return _blend(dry, shaped.astype(np.float32), amount * 0.85)


def _echo_reduce(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    processed = dry.astype(np.float64).copy()
    for delay_ms, weight in [(55, 0.18), (110, 0.12), (180, 0.08)]:
        shift = int(sr * delay_ms / 1000.0)
        if shift <= 0 or shift >= len(dry):
            continue
        delayed = np.zeros_like(processed)
        delayed[shift:] = dry[:-shift]
        processed -= delayed * (weight * (0.4 + amount * 0.9))
    processed = apply_limiter(processed.astype(np.float32), ceiling_db=-0.8)
    return _blend(dry, processed, amount)


def _reverb_reduce(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    kernel_len = max(8, int(sr * (0.01 + 0.05 * amount)))
    kernel = np.exp(-np.linspace(0.0, 4.0, kernel_len))
    kernel /= max(np.sum(kernel), 1e-6)

    def _channel_dereverb(channel: np.ndarray) -> np.ndarray:
        tail = fftconvolve(channel, kernel, mode="full")[: len(channel)]
        return channel - (tail - channel) * (0.35 + 0.45 * amount)

    wet = _process_each_channel(dry, _channel_dereverb)
    return _blend(dry, wet, amount)


def _noise_gate(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    mono = np.mean(np.abs(dry), axis=1)
    floor = np.percentile(mono, 35)
    threshold = floor + (np.percentile(mono, 75) - floor) * (0.08 + amount * 0.12)
    soft_width = max(threshold * 0.8, 1e-5)
    mask = np.clip((mono - threshold + soft_width) / (2 * soft_width), 0.0, 1.0)
    win = max(8, int(sr * 0.01))
    kernel = np.ones(win, dtype=np.float32) / win
    mask = np.convolve(mask, kernel, mode="same")
    min_gain = 1.0 - amount * 0.7
    gain = min_gain + (1.0 - min_gain) * mask
    wet = dry * gain[:, np.newaxis]
    return wet.astype(np.float32)


def _equalizer(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    wet = apply_eq(
        _ensure_2d(audio),
        sr,
        [
            EQBand(110, -1.5 * amount, 0.8),
            EQBand(2600, 4.0 * amount, 1.1),
            EQBand(9000, 2.2 * amount, 0.9),
        ],
    )
    return _blend(_ensure_2d(audio), wet, amount)


def _compressor(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    wet = apply_compressor(
        _ensure_2d(audio),
        sr,
        threshold_db=-26 + 12 * (1.0 - amount),
        ratio=1.5 + amount * 4.5,
        attack_ms=3.0 + amount * 6.0,
        release_ms=35.0 + amount * 80.0,
    )
    wet = normalize(wet, target_db=-1.5)
    return _blend(_ensure_2d(audio), wet, amount)


def _warmth(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    wet = apply_eq(
        _ensure_2d(audio),
        sr,
        [
            EQBand(180, 3.0 * amount, 0.9),
            EQBand(420, 1.6 * amount, 1.0),
            EQBand(6200, -1.4 * amount, 0.8),
        ],
    )
    return _blend(_ensure_2d(audio), wet, amount)


def _brightness(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    wet = apply_eq(
        _ensure_2d(audio),
        sr,
        [
            EQBand(180, -1.2 * amount, 0.8),
            EQBand(4200, 2.8 * amount, 1.0),
            EQBand(9500, 4.2 * amount, 0.7),
        ],
    )
    return _blend(_ensure_2d(audio), wet, amount)


def _stereo_width(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    wet = apply_stereo_widening(_ensure_2d(audio), width=1.0 + 0.9 * amount)
    return _blend(_ensure_2d(audio), wet, amount)


def _studio_sound(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    magnitude = abs(float(amount))
    result = _reverb_reduce(dry, sr, min(1.0, 0.35 + magnitude * 0.5))
    result = _equalizer(result, sr, 0.45 + magnitude * 0.35)
    result = _compressor(result, sr, 0.35 + magnitude * 0.45)
    result = _stereo_width(result, sr, magnitude * 0.25)
    result = normalize(result, target_db=-1.4)
    wet = apply_limiter(result, ceiling_db=-0.4)
    return _signed_delta_blend(dry, wet, amount)


def _vocal_presence(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    magnitude = abs(float(amount))
    result = apply_eq(
        dry,
        sr,
        [
            EQBand(180, -1.4 * magnitude, 0.8),
            EQBand(2500, 4.5 * magnitude, 1.2),
            EQBand(5200, 2.5 * magnitude, 0.9),
        ],
    )
    result = _blend(dry, result, min(1.0, 0.35 + magnitude * 0.35))
    result = _compressor(result, sr, 0.25 + magnitude * 0.5)
    wet = apply_limiter(result, ceiling_db=-0.5)
    return _signed_delta_blend(dry, wet, amount)


def _broadcast_ready(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    dry = _ensure_2d(audio)
    magnitude = abs(float(amount))
    result = _compressor(dry, sr, 0.45 + magnitude * 0.45)
    result = _equalizer(result, sr, 0.25 + magnitude * 0.25)
    result = normalize(result, target_db=-1.0)
    wet = apply_limiter(result, ceiling_db=-0.3)
    return _signed_delta_blend(dry, wet, amount)


def _gain(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    del sr
    dry = _ensure_2d(audio)
    db = float(np.clip(amount, -1.0, 1.0)) * 18.0
    gain = 10 ** (db / 20.0)
    wet = dry * gain
    return np.clip(wet, -1.0, 1.0).astype(np.float32)


EFFECT_PRESETS = {
    "Studio Sound": {
        "description": "Subtle compression + EQ polish + stereo widening",
        "eq": [EQBand(80, 2.0, 0.7), EQBand(3000, 1.5, 1.0), EQBand(10000, 2.0, 0.8)],
        "compressor": {"threshold_db": -18, "ratio": 2.5, "attack_ms": 10, "release_ms": 80},
        "stereo_width": 1.3,
    },
    "Vocal Presence": {
        "description": "Boost vocal clarity with mid-frequency EQ + compression",
        "eq": [EQBand(200, -2.0, 0.8), EQBand(2500, 4.0, 1.2), EQBand(5000, 2.0, 1.0)],
        "compressor": {"threshold_db": -15, "ratio": 3.0, "attack_ms": 5, "release_ms": 40},
        "stereo_width": 1.0,
    },
    "Broadcast Ready": {
        "description": "Normalized + compressed + limited for broadcast",
        "eq": [EQBand(80, -1.0, 0.5), EQBand(1000, 1.0, 0.7), EQBand(8000, 1.5, 0.8)],
        "compressor": {"threshold_db": -12, "ratio": 4.0, "attack_ms": 3, "release_ms": 30},
        "normalize": -1.0,
        "limiter": -0.3,
        "stereo_width": 1.0,
    },
    "Equalizer": {
        "description": "Custom 3-band EQ only",
        "eq": [EQBand(80, 0.0, 0.7), EQBand(2500, 0.0, 1.0), EQBand(10000, 0.0, 0.8)],
    },
    "Compressor": {
        "description": "Custom compressor only",
        "compressor": {"threshold_db": -20, "ratio": 4.0, "attack_ms": 5, "release_ms": 50},
    },
}


EFFECT_SPECS: list[EffectSpec] = [
    EffectSpec("reverb", "Reverb", "Sound Effects", "Add room ambience and tail to the sound.", 0.28),
    EffectSpec("delay", "Delay", "Sound Effects", "Add audible repeats and space.", 0.22),
    EffectSpec("echo_reduce", "Echo Reducer", "Repair & Cleanup", "Reduce slapback-style echo and short reflections.", 0.40),
    EffectSpec("reverb_reduce", "Reverb Remover", "Repair & Cleanup", "Tighten roomy recordings and reduce lingering ambience.", 0.45),
    EffectSpec("noise_gate", "Noise Gate", "Repair & Cleanup", "Reduce low-level room noise between phrases.", 0.30),
    EffectSpec("equalizer", "Equalizer", "Tone & Dynamics", "Shape bass, mids, and air for clarity.", 0.40),
    EffectSpec("compressor", "Compressor", "Tone & Dynamics", "Even out loud and soft parts for steadier vocals.", 0.35),
    EffectSpec("gain", "Gain / Trim", "Tone & Dynamics", "Boost or trim the layer level before the final clip guard.", 0.25, -1.0, 1.0),
    EffectSpec("warmth", "Warmth", "Tone & Dynamics", "Add low-mid body and tame harsh highs.", 0.28),
    EffectSpec("brightness", "Air / Brightness", "Tone & Dynamics", "Add clarity and top-end sparkle.", 0.30),
    EffectSpec("stereo_width", "Stereo Width", "Tone & Dynamics", "Widen stereo material without changing the dry center.", 0.25),
    EffectSpec("studio_sound", "Studio Sound", "Vocal Enhancers", "Polish vocals with cleanup, tone shaping, and loudness control. Negative values subtract the enhancer curve.", 0.45, -1.0, 1.0),
    EffectSpec("vocal_presence", "Vocal Presence", "Vocal Enhancers", "Bring the voice forward with presence EQ and compression. Negative values subtract the enhancer curve.", 0.40, -1.0, 1.0),
    EffectSpec("broadcast_ready", "Broadcast Ready", "Vocal Enhancers", "Tighter dynamics and loudness for spoken-word delivery. Negative values subtract the enhancer curve.", 0.42, -1.0, 1.0),
]

EFFECT_SPEC_MAP = {spec.key: spec for spec in EFFECT_SPECS}
EFFECT_SECTION_ORDER = ["Sound Effects", "Repair & Cleanup", "Tone & Dynamics", "Vocal Enhancers"]
EFFECT_KEYS_BY_SECTION = {
    section: [spec.key for spec in EFFECT_SPECS if spec.section == section]
    for section in EFFECT_SECTION_ORDER
}


def empty_effect_state() -> dict[str, dict[str, float | bool]]:
    return {
        spec.key: {"enabled": False, "amount": spec.default_amount}
        for spec in EFFECT_SPECS
    }


def normalize_effect_state(
    state: dict[str, dict[str, float | bool]] | None,
) -> dict[str, dict[str, float | bool]]:
    normalized = empty_effect_state()
    if not state:
        return normalized
    for key, spec_state in state.items():
        if key not in normalized or not isinstance(spec_state, dict):
            continue
        spec = EFFECT_SPEC_MAP[key]
        amount = float(spec_state.get("amount", normalized[key]["amount"]))
        normalized[key]["amount"] = float(np.clip(amount, spec.min_amount, spec.max_amount))
        normalized[key]["enabled"] = bool(spec_state.get("enabled", False))
    return normalized


def apply_preset(
    audio: np.ndarray,
    sr: int,
    preset_name: str,
    custom_eq: list[EQBand] | None = None,
    custom_comp: dict | None = None,
) -> np.ndarray:
    preset = EFFECT_PRESETS.get(preset_name)
    if not preset:
        return _ensure_2d(audio)
    result = _ensure_2d(audio).copy()

    eq_bands = custom_eq if custom_eq is not None else preset.get("eq")
    if eq_bands:
        result = apply_eq(result, sr, eq_bands)

    comp_params = custom_comp if custom_comp is not None else preset.get("compressor")
    if comp_params:
        result = apply_compressor(result, sr, **comp_params)

    if "stereo_width" in preset:
        result = apply_stereo_widening(result, preset["stereo_width"])
    if "normalize" in preset:
        result = normalize(result, preset["normalize"])
    if "limiter" in preset:
        result = apply_limiter(result, preset["limiter"])

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def apply_effect_stack(
    audio: np.ndarray,
    sr: int,
    effect_state: dict[str, dict[str, float | bool]] | None,
) -> np.ndarray:
    result = _ensure_2d(audio).astype(np.float32)
    state = normalize_effect_state(effect_state)
    processors = {
        "noise_gate": _noise_gate,
        "echo_reduce": _echo_reduce,
        "reverb_reduce": _reverb_reduce,
        "equalizer": _equalizer,
        "warmth": _warmth,
        "brightness": _brightness,
        "compressor": _compressor,
        "gain": _gain,
        "stereo_width": _stereo_width,
        "studio_sound": _studio_sound,
        "vocal_presence": _vocal_presence,
        "broadcast_ready": _broadcast_ready,
        "delay": _simple_delay,
        "reverb": _simple_reverb,
    }
    order = [
        "noise_gate",
        "echo_reduce",
        "reverb_reduce",
        "equalizer",
        "warmth",
        "brightness",
        "compressor",
        "gain",
        "stereo_width",
        "studio_sound",
        "vocal_presence",
        "broadcast_ready",
        "delay",
        "reverb",
    ]
    for key in order:
        entry = state.get(key)
        if not entry or not entry["enabled"]:
            continue
        spec = EFFECT_SPEC_MAP[key]
        amount = float(np.clip(entry["amount"], spec.min_amount, spec.max_amount))
        if abs(amount) <= 1e-6:
            continue
        result = processors[key](result, sr, amount)
        result = np.clip(result, -1.0, 1.0).astype(np.float32)
    return result
