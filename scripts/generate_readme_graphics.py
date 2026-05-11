"""Generate branded README graphics for AI Audio Toolkit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "assets" / "readme"
WIDTH = 1600
HEIGHT = 900
BRAND = "AI Audio Toolkit"


BG_TOP = "#101722"
BG_BOTTOM = "#18253A"
PANEL = "#172033"
PANEL_2 = "#1E2B44"
TEXT = "#F5FAFF"
MUTED = "#AAB6C8"
TEAL = "#24C7BD"
BLUE = "#6AA6FF"
ORANGE = "#F59F3D"
GREEN = "#75D69C"
RED = "#FF7A72"


@dataclass(frozen=True)
class FontSet:
    title: ImageFont.FreeTypeFont
    h1: ImageFont.FreeTypeFont
    h2: ImageFont.FreeTypeFont
    h3: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont


def _font_path(*names: str) -> str | None:
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/System/Library/Fonts"),
    ]
    for font_dir in font_dirs:
        for name in names:
            path = font_dir / name
            if path.is_file():
                return str(path)
    return None


def _load_font(name: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if name:
        return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def _fonts() -> FontSet:
    regular = _font_path("segoeui.ttf", "DejaVuSans.ttf", "Helvetica.ttc")
    semibold = _font_path("seguisb.ttf", "SegoeUI-SemiBold.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc")
    bold = _font_path("segoeuib.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc")
    mono = _font_path("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc")
    return FontSet(
        title=_load_font(bold, 76),
        h1=_load_font(bold, 58),
        h2=_load_font(semibold, 38),
        h3=_load_font(semibold, 28),
        body=_load_font(regular, 24),
        small=_load_font(regular, 20),
        mono=_load_font(mono, 22),
    )


FONTS = _fonts()


def _hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _gradient(size: tuple[int, int], top: str, bottom: str) -> Image.Image:
    w, h = size
    top_rgb = _hex(top)
    bottom_rgb = _hex(bottom)
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        color = tuple(_lerp(top_rgb[i], bottom_rgb[i], t) for i in range(3))
        for x in range(w):
            px[x, y] = color
    return img


def _background() -> Image.Image:
    img = _gradient((WIDTH, HEIGHT), BG_TOP, BG_BOTTOM).convert("RGBA")
    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    draw.ellipse((-260, -220, 620, 520), fill=(36, 199, 189, 48))
    draw.ellipse((1010, 500, 1900, 1220), fill=(245, 159, 61, 46))
    draw.ellipse((920, -300, 1780, 440), fill=(106, 166, 255, 34))
    glow = glow.filter(ImageFilter.GaussianBlur(70))
    img.alpha_composite(glow)
    return img


def _round_rect(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=1, radius=28):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=TEXT, anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _pill(draw: ImageDraw.ImageDraw, xy, label: str, color: str, font=None):
    font = font or FONTS.small
    x, y = xy
    bbox = draw.textbbox((0, 0), label, font=font)
    w = bbox[2] - bbox[0] + 38
    h = bbox[3] - bbox[1] + 22
    _round_rect(draw, (x, y, x + w, y + h), fill="#21314D", outline=color, width=2, radius=h // 2)
    draw.ellipse((x + 14, y + h // 2 - 5, x + 24, y + h // 2 + 5), fill=color)
    _text(draw, (x + 34, y + 10), label, font, fill=TEXT)
    return x + w + 12


def _wrap(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def _paragraph(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, width: int, leading: int):
    x, y = xy
    for line in _wrap(text, font, width):
        _text(draw, (x, y), line, font, fill=fill)
        y += leading
    return y


def _waveform(draw: ImageDraw.ImageDraw, box, color=TEAL):
    x1, y1, x2, y2 = box
    mid = (y1 + y2) // 2
    points_top = []
    samples = [0.08, 0.22, 0.11, 0.48, 0.18, 0.38, 0.13, 0.58, 0.26, 0.46, 0.14, 0.32, 0.08]
    step = (x2 - x1) / (len(samples) - 1)
    for i, amp in enumerate(samples):
        x = int(x1 + i * step)
        points_top.append((x, int(mid - amp * (y2 - y1))))
    points_bottom = [(x, int(mid + (mid - y) * 0.82)) for x, y in reversed(points_top)]
    draw.polygon(points_top + points_bottom, fill=color + "88")
    draw.line([(x1, mid), (x2, mid)], fill="#6E7C92", width=2)


def _track(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, color: str, width: int = 610):
    _round_rect(draw, (x, y, x + width, y + 72), fill="#121A2A", outline="#2B3A56", radius=18)
    draw.ellipse((x + 22, y + 22, x + 50, y + 50), fill=color)
    _text(draw, (x + 66, y + 22), label, FONTS.small, fill=TEXT)
    _waveform(draw, (x + 210, y + 18, x + width - 26, y + 54), color)


def hero() -> Image.Image:
    img = _background()
    draw = ImageDraw.Draw(img)

    _text(draw, (92, 88), BRAND, FONTS.title)
    _paragraph(
        draw,
        (98, 194),
        "AI stem separation and speech repair in one desktop workspace.",
        FONTS.h2,
        "#DCEAFF",
        680,
        48,
    )

    y = 330
    x = 98
    max_x = 704
    for label, color in [
        ("audio-separator presets", TEAL),
        ("SAM-Audio prompting", BLUE),
        ("speech enhancement", ORANGE),
    ]:
        label_width = draw.textbbox((0, 0), label, font=FONTS.small)[2] + 38
        if x + label_width > max_x:
            x = 98
            y += 64
        x = _pill(draw, (x, y), label, color)

    workflow = [
        ("1", "Load audio or video", "Open music, interviews, podcasts, town halls, or video files."),
        ("2", "Choose the target", "Split vocals and instruments, isolate prompted sounds, or clean speech."),
        ("3", "Use the result", "Review cleaned speech, karaoke mixes, or separated stems."),
    ]
    y = 462
    for number, title, body in workflow:
        _round_rect(draw, (98, y, 680, y + 112), fill="#162236", outline="#2D405F", radius=24)
        draw.ellipse((128, y + 28, 184, y + 84), fill="#223D5E", outline=TEAL, width=2)
        _text(draw, (156, y + 42), number, FONTS.h3, anchor="mm")
        _text(draw, (210, y + 24), title, FONTS.h3)
        _paragraph(draw, (210, y + 62), body, FONTS.small, MUTED, 410, 26)
        y += 132

    _round_rect(draw, (760, 90, 1508, 810), fill=PANEL, outline="#344765", width=2, radius=34)
    _round_rect(draw, (790, 122, 1478, 188), fill="#111827", outline="#2F425D", radius=20)
    _text(draw, (820, 140), "Separation Session", FONTS.h3)
    _pill(draw, (1190, 132), "GPU ready", GREEN, FONTS.small)

    _round_rect(draw, (790, 224, 1478, 408), fill="#101827", outline="#263951", radius=24)
    _text(draw, (820, 250), "input_mix.wav", FONTS.small, MUTED)
    _waveform(draw, (826, 294, 1442, 368), TEAL)

    y = 446
    for label, color in [
        ("Vocals", TEAL),
        ("Drums", ORANGE),
        ("Bass", BLUE),
        ("Speech cleaned", GREEN),
    ]:
        _track(draw, 790, y, label, color, width=688)
        y += 88

    _round_rect(draw, (912, 740, 1356, 786), fill="#20304B", outline="#3D536F", radius=23)
    _text(draw, (1134, 751), "Export cleaned or separated audio", FONTS.small, anchor="ma")
    return img.convert("RGB")


def _metric_card(draw, box, title: str, body: str, color: str):
    _round_rect(draw, box, fill=PANEL, outline="#334763", width=2, radius=26)
    x1, y1, x2, _ = box
    draw.ellipse((x1 + 28, y1 + 28, x1 + 64, y1 + 64), fill=color)
    _text(draw, (x1 + 84, y1 + 24), title, FONTS.h3)
    _paragraph(draw, (x1 + 30, y1 + 86), body, FONTS.small, MUTED, x2 - x1 - 60, 28)


def enhancement_engines() -> Image.Image:
    img = _background()
    draw = ImageDraw.Draw(img)
    _text(draw, (82, 72), "Enhancement engines", FONTS.h1)
    _text(draw, (84, 142), BRAND, FONTS.h3, fill=TEAL)
    _paragraph(
        draw,
        (84, 190),
        "Repair spoken-word recordings with dedicated engines for noise, room echo, clipping, reverb, and low-quality audio.",
        FONTS.body,
        "#DCEAFF",
        760,
        34,
    )

    cards = [
        ((84, 300, 480, 506), "Studio Sound", "One-pass DeepFilterNet cleanup plus Roformer de-reverb controls.", TEAL),
        ((520, 300, 916, 506), "ClearVoice", "Town halls, distant mics, conference rooms, and HVAC-heavy captures.", BLUE),
        ((956, 300, 1352, 506), "VoiceFixer", "Heavy restoration with restoration modes and long-file stitching.", ORANGE),
        ((84, 548, 480, 754), "MetricGAN+", "Noise suppression with automatic sample-rate round trip.", GREEN),
        ((520, 548, 916, 754), "MDX-Net", "Pull speech or vocals out of dense foreground and background mixes.", RED),
        ((956, 548, 1352, 754), "Output leveling", "Match input loudness, auto level speech, trim gain, and apply limiter.", "#B99BFF"),
    ]
    for box, title, body, color in cards:
        _metric_card(draw, box, title, body, color)

    _round_rect(draw, (1262, 84, 1510, 210), fill="#111827", outline="#334763", radius=26)
    _text(draw, (1290, 108), "Shared controls", FONTS.h3)
    _text(draw, (1290, 150), "segment size", FONTS.small, MUTED)
    _text(draw, (1290, 178), "overlap + batch", FONTS.small, MUTED)
    return img.convert("RGB")


def model_library() -> Image.Image:
    img = _background()
    draw = ImageDraw.Draw(img)
    _text(draw, (82, 72), "Model library", FONTS.h1)
    _text(draw, (84, 142), BRAND, FONTS.h3, fill=TEAL)
    _paragraph(
        draw,
        (84, 190),
        "Pick a model by job: vocals, karaoke, full-band stems, individual instruments, cleanup, or prompt-based isolation.",
        FONTS.body,
        "#DCEAFF",
        790,
        34,
    )

    groups = [
        ("Vocal separation", ["BS-Roformer", "Kim Vocal 2", "InstVoc HQ", "Kuielab Vocals"], TEAL),
        ("Karaoke", ["MelBand Roformer", "MDX-NET Karaoke 2", "6-HP Karaoke"], ORANGE),
        ("Multi-stem band", ["HTDemucs Fine-Tuned", "HTDemucs 6-Stem", "HDemucs MMI"], BLUE),
        ("Cleanup", ["Denoise Roformer", "UVR DeNoise", "De-Echo/DeReverb"], GREEN),
    ]

    x_positions = [84, 458, 832, 1206]
    for x, (title, items, color) in zip(x_positions, groups):
        _round_rect(draw, (x, 310, x + 310, 758), fill=PANEL, outline="#334763", radius=26)
        draw.rectangle((x + 30, 346, x + 280, 354), fill=color)
        _text(draw, (x + 30, 382), title, FONTS.h3)
        y = 450
        for item in items:
            _round_rect(draw, (x + 28, y, x + 282, y + 58), fill="#111827", outline="#2D3D57", radius=18)
            draw.ellipse((x + 48, y + 21, x + 64, y + 37), fill=color)
            _text(draw, (x + 78, y + 16), item, FONTS.small)
            y += 76

    _round_rect(draw, (84, 790, 1516, 842), fill="#142137", outline="#334763", radius=22)
    x = 112
    for label, color in [
        ("Roformer", TEAL),
        ("MDX-Net", BLUE),
        ("MDXC", ORANGE),
        ("VR", GREEN),
        ("Demucs", "#B99BFF"),
        ("SAM-Audio", RED),
    ]:
        x = _pill(draw, (x, 797), label, color, FONTS.small)
    return img.convert("RGB")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ai-audio-toolkit-hero.png": hero(),
        "enhancement-engines.png": enhancement_engines(),
        "model-library.png": model_library(),
    }
    for filename, image in outputs.items():
        path = OUTPUT_DIR / filename
        image.save(path, quality=95)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
