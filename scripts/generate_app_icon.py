"""Generate the ai-audio-toolkit application icon assets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


SIZE = 1024
CANVAS_PADDING = 44
TILE_RADIUS = 214
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets"
ICON_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _linear_gradient(size: int, start: str, end: str, angle_deg: float = 120.0) -> Image.Image:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    angle = np.deg2rad(angle_deg)
    axis = np.cos(angle) * xx + np.sin(angle) * yy
    axis -= axis.min()
    axis /= axis.max()
    gradient = start_rgb * (1.0 - axis[..., None]) + end_rgb * axis[..., None]
    return Image.fromarray(np.clip(gradient, 0, 255).astype(np.uint8), mode="RGB")


def _radial_glow(size: int, center: tuple[float, float], radius: float, color: str, alpha: int) -> Image.Image:
    cx, cy = center
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = _hex_to_rgb(color).astype(np.uint8)
    rgba[..., 3] = np.clip(glow * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _rounded_rectangle_mask(size: int, box: tuple[int, int, int, int], radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(box, radius=radius, fill=255)
    return mask


def _trim_transparent_padding(image: Image.Image, padding: int = 10) -> Image.Image:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    cropped = image.crop((
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    ))
    return cropped.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def _draw_waveform(draw: ImageDraw.ImageDraw, color: str, points: list[tuple[int, int]], width: int) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    radius = width // 2
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def build_icon() -> Image.Image:
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    tile_box = (
        CANVAS_PADDING,
        CANVAS_PADDING,
        SIZE - CANVAS_PADDING,
        SIZE - CANVAS_PADDING,
    )
    tile_mask = _rounded_rectangle_mask(SIZE, tile_box, TILE_RADIUS)

    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_box = (
        tile_box[0],
        tile_box[1] + 18,
        tile_box[2],
        tile_box[3] + 18,
    )
    shadow_draw.rounded_rectangle(shadow_box, radius=TILE_RADIUS, fill=(6, 10, 22, 145))
    shadow = shadow.filter(ImageFilter.GaussianBlur(42))
    canvas.alpha_composite(shadow)

    tile = _linear_gradient(SIZE, "#0F2746", "#28C7BD", angle_deg=120.0).convert("RGBA")
    tile.putalpha(tile_mask)
    canvas.alpha_composite(tile)

    tile_overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    tile_overlay.alpha_composite(_radial_glow(SIZE, (248, 232), 380, "#96FFF3", 64))
    tile_overlay.alpha_composite(_radial_glow(SIZE, (796, 792), 440, "#0E5FFF", 56))
    tile_overlay.putalpha(ImageChops.multiply(tile_overlay.getchannel("A"), tile_mask))
    canvas.alpha_composite(tile_overlay)

    interior = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    interior_draw = ImageDraw.Draw(interior)
    interior_draw.rounded_rectangle(
        (148, 148, SIZE - 148, SIZE - 148),
        radius=168,
        outline=(255, 255, 255, 26),
        width=4,
    )
    canvas.alpha_composite(interior)

    waveform_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    waveform_draw = ImageDraw.Draw(waveform_layer)
    left_points = [
        (186, 580),
        (262, 580),
        (328, 404),
        (398, 640),
        (472, 448),
        (512, 512),
    ]
    right_points = [
        (512, 512),
        (562, 420),
        (636, 672),
        (708, 362),
        (780, 580),
        (848, 580),
    ]
    _draw_waveform(waveform_draw, "#F4FAFF", left_points, 88)
    _draw_waveform(waveform_draw, "#D3F9F6", right_points, 88)
    waveform_layer = waveform_layer.filter(ImageFilter.GaussianBlur(0.4))
    canvas.alpha_composite(waveform_layer)

    stem_shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    stem_shadow_draw = ImageDraw.Draw(stem_shadow)
    stem_shadow_draw.rounded_rectangle((468, 228, 560, 808), radius=44, fill=(10, 19, 35, 110))
    stem_shadow = stem_shadow.filter(ImageFilter.GaussianBlur(18))
    canvas.alpha_composite(stem_shadow)

    stem_gradient = _linear_gradient(SIZE, "#FFC955", "#FF6A4D", angle_deg=102.0).convert("RGBA")
    stem_mask = Image.new("L", (SIZE, SIZE), 0)
    stem_mask_draw = ImageDraw.Draw(stem_mask)
    stem_mask_draw.rounded_rectangle((468, 216, 560, 796), radius=44, fill=255)
    stem_gradient.putalpha(stem_mask)
    canvas.alpha_composite(stem_gradient)

    handle = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    handle_draw = ImageDraw.Draw(handle)
    handle_draw.rounded_rectangle((430, 438, 598, 586), radius=74, fill=(255, 247, 232, 232))
    handle_draw.rounded_rectangle((454, 462, 574, 562), radius=50, fill=(255, 181, 82, 245))
    canvas.alpha_composite(handle.filter(ImageFilter.GaussianBlur(0.6)))

    accent = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    accent_draw = ImageDraw.Draw(accent)
    accent_draw.rounded_rectangle((156, 190, 328, 316), radius=56, fill=(232, 252, 250, 44))
    accent_draw.rounded_rectangle((168, 202, 314, 300), radius=48, fill=(255, 255, 255, 30))
    canvas.alpha_composite(accent)

    return _trim_transparent_padding(canvas, padding=6)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    icon = build_icon()
    png_path = OUTPUT_DIR / "app_icon.png"
    ico_path = OUTPUT_DIR / "app_icon.ico"
    icon.save(png_path)
    icon.save(
        ico_path,
        sizes=[(size, size) for size in ICON_SIZES],
        bitmap_format="bmp",
    )
    print(f"Saved {png_path}")
    print(f"Saved {ico_path}")


if __name__ == "__main__":
    main()
