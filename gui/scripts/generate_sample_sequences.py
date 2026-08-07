"""Generate the bundled "Try a sample" frame sequences (roadmap Phase 6.3,
issue #17: "Bundled sample projects").

Produces small, entirely synthetic, procedurally-drawn vertical-scroll
sequences under ``gui/resources/samples/<name>/frame_NN.png`` so a new user
can explore every ``HybridStitchPanel`` tool tab immediately, without any
real (and, per this repo's benchmark corpus, potentially sensitive) source
art.

Each sample is built by drawing one tall synthetic "page" -- flat-color
panels, simple line art, gradients, and text labels, all generated with
PIL's drawing primitives (mirroring the ``np.zeros``/``cv2.rectangle``-style
synthetic fixtures already used across ``backend/test/`` and ``gui/test/``,
just with PIL since that's this package's own image-handling dependency) --
then slicing it into overlapping vertical crops that simulate a scrolling
capture. Consecutive frames share real, spatially-consistent overlap, so the
stitching pipeline's basic mechanics (frame-pair alignment, seam blending)
have something genuine to exercise.

Run directly to (re)generate the bundled assets:

    python -m gui.scripts.generate_sample_sequences
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "resources" / "samples"

# Frame geometry: small enough that the whole bundle stays a few hundred KB.
_FRAME_W = 480
_FRAME_H = 320
_OVERLAP_FRAC = 0.35  # fraction of a frame's height shared with the next one
_STRIDE = int(_FRAME_H * (1 - _OVERLAP_FRAC))
_N_FRAMES = 6


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _draw_gradient_band(
    draw: ImageDraw.ImageDraw,
    width: int,
    y0: int,
    y1: int,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> None:
    for y in range(y0, y1):
        t = (y - y0) / max(1, (y1 - y0 - 1))
        draw.line([(0, y), (width, y)], fill=_lerp(top, bottom, t))


def _draw_page_scroll_a(width: int, height: int) -> Image.Image:
    """Cool-toned gradient page with a few flat "panel" shapes -- exercises
    smooth photometric blending across the seam."""
    img = Image.new("RGB", (width, height), (20, 24, 32))
    draw = ImageDraw.Draw(img)
    _draw_gradient_band(draw, width, 0, height, (24, 40, 64), (10, 14, 24))

    # A handful of flat rounded panels down the page, alternating color, so
    # every frame slice has distinctive, well-separated edges to match on.
    n_panels = 9
    panel_h = height // n_panels
    palette = [(235, 180, 60), (90, 200, 210), (220, 90, 110), (140, 210, 100)]
    for i in range(n_panels):
        y0 = i * panel_h + panel_h // 6
        y1 = (i + 1) * panel_h - panel_h // 6
        x0 = width // 8 + (i % 2) * width // 10
        x1 = width - width // 8 - ((i + 1) % 2) * width // 10
        color = palette[i % len(palette)]
        draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=color)
        draw.text((x0 + 10, y0 + 8), f"PANEL {i + 1:02d}", fill=(20, 20, 20))
    return img


def _draw_page_scroll_b(width: int, height: int) -> Image.Image:
    """Simple line-art page: a wavy vertical "character silhouette" outline
    plus horizontal rule lines -- exercises seam placement along line art
    rather than through flat fill."""
    img = Image.new("RGB", (width, height), (245, 244, 240))
    draw = ImageDraw.Draw(img)

    # Horizontal rule lines every ~40px, like panel borders on a manga page.
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=(210, 208, 200), width=2)

    # A wavy outline running the length of the page, thick enough to survive
    # downscaling and give the matcher stable line-art edges.
    cx = width // 2
    amp = width // 6
    points = []
    for y in range(0, height, 4):
        x = cx + int(amp * math.sin(y / 90.0)) + int(amp * 0.4 * math.sin(y / 27.0 + 1.3))
        points.append((x, y))
    draw.line(points, fill=(40, 40, 45), width=6, joint="curve")

    # A few flat "speech bubble" ellipses with labels to add distinct
    # keypoints away from the central line.
    for i, y in enumerate(range(60, height - 60, 260)):
        bx = width - 140 if i % 2 == 0 else 40
        draw.ellipse([bx, y, bx + 100, y + 60], outline=(40, 40, 45), width=3, fill=(255, 255, 255))
        draw.text((bx + 12, y + 22), f"p.{i + 1}", fill=(40, 40, 45))
    return img


def _draw_page_scroll_c(width: int, height: int) -> Image.Image:
    """Warm gradient page with a grid of small marks and text labels --
    exercises frame-index/text-heavy scroll content."""
    img = Image.new("RGB", (width, height), (30, 20, 20))
    draw = ImageDraw.Draw(img)
    _draw_gradient_band(draw, width, 0, height, (60, 30, 20), (15, 10, 12))

    # Dot grid for extra distinctive keypoints across otherwise flat areas.
    for y in range(20, height, 30):
        for x in range(20, width, 30):
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(200, 150, 90))

    n_labels = 10
    step = height // n_labels
    for i in range(n_labels):
        y = i * step + step // 3
        draw.rectangle([30, y, width - 30, y + 34], outline=(230, 190, 120), width=2)
        draw.text((44, y + 8), f"SAMPLE TEXT LINE {i + 1:02d}", fill=(230, 190, 120))
    return img


_PAGE_BUILDERS = {
    "scroll_a_panels": _draw_page_scroll_a,
    "scroll_b_lineart": _draw_page_scroll_b,
    "scroll_c_textgrid": _draw_page_scroll_c,
}


def _page_height(n_frames: int, frame_h: int, stride: int) -> int:
    return frame_h + stride * (n_frames - 1)


def generate_sequence(
    name: str,
    out_dir: Path,
    n_frames: int = _N_FRAMES,
    frame_w: int = _FRAME_W,
    frame_h: int = _FRAME_H,
    stride: int = _STRIDE,
) -> list[Path]:
    """Build one synthetic scroll page and slice it into overlapping frames,
    written to ``out_dir/frame_NN.png``. Returns the written paths in
    scroll order."""
    builder = _PAGE_BUILDERS[name]
    page_h = _page_height(n_frames, frame_h, stride)
    page = builder(frame_w, page_h)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_frames):
        y0 = i * stride
        y1 = y0 + frame_h
        crop = page.crop((0, y0, frame_w, y1))
        path = out_dir / f"frame_{i:02d}.png"
        crop.save(path, format="PNG", optimize=True)
        paths.append(path)
    return paths


def generate_all(dest: Path = SAMPLES_DIR) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for name in _PAGE_BUILDERS:
        result[name] = generate_sequence(name, dest / name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=SAMPLES_DIR,
        help="Directory to write sample sequences under (default: gui/resources/samples).",
    )
    args = parser.parse_args()
    written = generate_all(args.dest)
    total_bytes = sum(p.stat().st_size for paths in written.values() for p in paths)
    for name, paths in written.items():
        print(f"{name}: {len(paths)} frames -> {paths[0].parent}")
    print(f"Total size: {total_bytes / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
