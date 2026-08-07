"""Background-pixel sampling strategies for bundle-adjustment anchor points."""

from __future__ import annotations

import numpy as np


def _sample_bg_points(
    mask: np.ndarray | None, H: int, W: int, n: int = 200
) -> np.ndarray:
    """Sample up to n (x,y) pixel coordinates from the background mask."""
    if mask is None:
        ys = np.random.randint(0, H, n)
        xs = np.random.randint(0, W, n)
    else:
        ys_bg, xs_bg = np.where(mask > 0)
        if len(ys_bg) == 0:
            ys = np.random.randint(0, H, n)
            xs = np.random.randint(0, W, n)
        else:
            idx = np.random.choice(len(ys_bg), min(n, len(ys_bg)), replace=False)
            ys, xs = ys_bg[idx], xs_bg[idx]
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _sample_bg_points_grid(
    mask: np.ndarray | None,
    H: int,
    W: int,
    n: int = 50,
    grid: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """
    Spatially-distributed background point sampler.

    Divides the image into a grid and draws points from each cell, ensuring
    coverage across all quadrants.  Non-LoFTR fallback edges use this instead
    of random sampling (P1.5 — W7 fix) so the BA solver receives spatially
    distributed anchor points rather than centre-biased random ones.
    """
    gr, gc = grid
    pts_list: list[np.ndarray] = []
    per_cell = max(1, n // (gr * gc))

    for r in range(gr):
        for c in range(gc):
            y0 = r * H // gr
            y1 = (r + 1) * H // gr
            x0 = c * W // gc
            x1 = (c + 1) * W // gc

            if mask is None:
                ys = np.random.randint(y0, max(y0 + 1, y1), per_cell)
                xs = np.random.randint(x0, max(x0 + 1, x1), per_cell)
            else:
                cell_mask = mask[y0:y1, x0:x1]
                ys_bg, xs_bg = np.where(cell_mask > 0)
                if len(ys_bg) == 0:
                    ys = np.random.randint(y0, max(y0 + 1, y1), per_cell)
                    xs = np.random.randint(x0, max(x0 + 1, x1), per_cell)
                else:
                    idx = np.random.choice(
                        len(ys_bg), min(per_cell, len(ys_bg)), replace=False
                    )
                    ys = ys_bg[idx] + y0
                    xs = xs_bg[idx] + x0

            pts_list.append(np.stack([xs, ys], axis=1).astype(np.float32))

    if not pts_list:
        return _sample_bg_points(mask, H, W, n)
    return np.concatenate(pts_list, axis=0)


__all__ = ["_sample_bg_points", "_sample_bg_points_grid"]
