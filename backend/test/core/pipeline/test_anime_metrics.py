from __future__ import annotations

import numpy as np
import pytest

from asp_backend.core.pipeline.anime_metrics import (
    cel_flatness_variance,
    extract_flat_cel_mask,
    extract_line_art,
    flat_region_edge_leakage,
    line_art_fracture_score,
)


def _create_synthetic_anime_frame(
    h: int = 200,
    w: int = 200,
    *,
    broken_line: bool = False,
    noisy_cel: bool = False,
) -> np.ndarray:
    """Create a synthetic anime frame with flat cel skin/bg and ink line art."""
    img = np.full((h, w, 3), 220, dtype=np.uint8)  # Flat skin tone
    
    # Add flat clothing region
    img[100:200, 30:170] = (80, 50, 180)  # Purple uniform

    if noisy_cel:
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw sharp ink outlines (black)
    if not broken_line:
        # Clean continuous box & diagonal line
        img[30:180, 30:33] = 0
        img[30:180, 167:170] = 0
        img[30:33, 30:170] = 0
        img[177:180, 30:170] = 0
        for i in range(50, 150):
            img[i, i : i + 2] = 0
    else:
        # Broken / shattered outlines with dashed gaps and isolated dots
        for y in range(30, 180, 10):
            img[y : y + 4, 30:33] = 0
            img[y : y + 4, 167:170] = 0
        for x in range(30, 170, 10):
            img[30:33, x : x + 4] = 0
            img[177:180, x : x + 4] = 0
        for i in range(50, 150, 10):
            img[i : i + 4, i : i + 2] = 0

    return img


def test_extract_line_art_and_flat_mask():
    clean_img = _create_synthetic_anime_frame()
    lines = extract_line_art(clean_img)
    assert lines.shape == (200, 200)
    assert lines.dtype == np.uint8
    assert lines.sum() > 0  # Detected ink lines

    flat_mask = extract_flat_cel_mask(clean_img, lines)
    assert flat_mask.shape == (200, 200)
    assert flat_mask.dtype == bool
    assert flat_mask.sum() > 1000  # Detected interior cel fill
    # Line pixels should not be in the flat mask
    assert (flat_mask & (lines > 0)).sum() == 0


def test_line_art_fracture_score_penalizes_broken_lines():
    clean_img = _create_synthetic_anime_frame(broken_line=False)
    broken_img = _create_synthetic_anime_frame(broken_line=True)

    score_clean = line_art_fracture_score(clean_img)
    score_broken = line_art_fracture_score(broken_img)

    # Broken lines must have significantly higher fracture score than continuous lines
    assert score_broken > score_clean
    assert score_clean < 25.0
    assert score_broken > 40.0


def test_cel_flatness_variance_penalizes_noisy_banding():
    clean_img = _create_synthetic_anime_frame(noisy_cel=False)
    noisy_img = _create_synthetic_anime_frame(noisy_cel=True)

    var_clean = cel_flatness_variance(clean_img)
    var_noisy = cel_flatness_variance(noisy_img)

    assert var_noisy > var_clean
    assert var_clean < 5.0


def test_flat_region_edge_leakage():
    clean_img = _create_synthetic_anime_frame(noisy_cel=False)
    noisy_img = _create_synthetic_anime_frame(noisy_cel=True)

    leak_clean = flat_region_edge_leakage(clean_img)
    leak_noisy = flat_region_edge_leakage(noisy_img)

    assert leak_noisy > leak_clean
