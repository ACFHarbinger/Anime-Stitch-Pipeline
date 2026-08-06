"""
ECC sub-pixel refinement of bundle-adjusted affines.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from backend.src.animation.core.stateless import _luma
from backend.src.constants import (
    ECC_EPS,
    ECC_MAX_DRIFT,
    ECC_MAX_ITER,
    ECC_PYRAMID_LEVELS,
)

try:
    import base as _batch_ecc
    _BATCH_ECC = (
        getattr(_batch_ecc, "__file__", None) is not None
        and hasattr(_batch_ecc, "fg_register")
        and hasattr(_batch_ecc.fg_register, "ecc_refine")
    )
    if not _BATCH_ECC:
        raise ImportError("compiled base.fg_register.ecc_refine extension not available")
except ImportError:
    _batch_ecc = None  # type: ignore[assignment]
    _BATCH_ECC = False


def _ecc_refine(
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
) -> List[np.ndarray]:
    """
    Sub-pixel refinement of each frame's affine using ECC maximisation.

    Motion model choice
    -------------------
    MOTION_AFFINE (6 DoF) is overpowered for pure panning shots and lets
    ECC drift in x when motion is predominantly vertical.  We use
    MOTION_TRANSLATION (2 DoF) instead, which is sufficient for anime pans
    and keeps the optimisation well-conditioned.

    Safety clamp
    ------------
    After ECC we reject any correction that moves dx or dy more than
    ECC_MAX_DRIFT pixels away from the bundle-adjusted starting point,
    falling back to the BA result.  This prevents ECC from diverging on
    low-texture frames.
    """

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITER,
        ECC_EPS,
    )
    refined = [affines[0].copy()]

    for i in range(1, len(frames)):
        M_i = affines[i].copy()  # global t_i
        M_prev = affines[i - 1].copy()  # global t_{i-1}

        ref_img = _luma(frames[i - 1])
        src_img = _luma(frames[i])
        ecc_mask = bg_masks[i - 1] if bg_masks[i - 1] is not None else None

        tx_rel_init = M_prev[0, 2] - M_i[0, 2]
        ty_rel_init = M_prev[1, 2] - M_i[1, 2]

        M_rel = np.eye(2, 3, dtype=np.float32)
        M_rel[0, 2] = tx_rel_init
        M_rel[1, 2] = ty_rel_init

        try:
            M_cur = M_rel.copy()
            for lvl in range(ECC_PYRAMID_LEVELS - 1, -1, -1):
                scale = 2**lvl
                r_s = cv2.resize(
                    ref_img,
                    (ref_img.shape[1] // scale, ref_img.shape[0] // scale),
                    interpolation=cv2.INTER_AREA,
                )
                s_s = cv2.resize(
                    src_img,
                    (src_img.shape[1] // scale, src_img.shape[0] // scale),
                    interpolation=cv2.INTER_AREA,
                )
                M_s = M_cur.copy()
                M_s[0, 2] /= scale
                M_s[1, 2] /= scale

                ecc_m_s = None
                if ecc_mask is not None:
                    ecc_m_s = cv2.resize(
                        ecc_mask,
                        (ecc_mask.shape[1] // scale, ecc_mask.shape[0] // scale),
                        interpolation=cv2.INTER_NEAREST,
                    )

                try:
                    if _BATCH_ECC:
                        M_s = _batch_ecc.fg_register.ecc_refine(
                            r_s, s_s, M_s, ecc_m_s,
                            cv2.MOTION_TRANSLATION,
                            ECC_MAX_ITER, ECC_EPS,
                        )
                    else:
                        _, M_s = cv2.findTransformECC(
                            r_s, s_s, M_s,
                            cv2.MOTION_TRANSLATION,
                            criteria,
                            ecc_m_s,
                            gaussFiltSize=5,
                        )
                    M_s[0, 2] *= scale
                    M_s[1, 2] *= scale
                    M_cur = M_s
                except (cv2.error, RuntimeError):
                    break

            tx_rel_ec, ty_rel_ec = M_cur[0, 2], M_cur[1, 2]
            tx_i_new = M_prev[0, 2] - tx_rel_ec
            ty_i_new = M_prev[1, 2] - ty_rel_ec

            if (
                abs(tx_i_new - M_i[0, 2]) > ECC_MAX_DRIFT
                or abs(ty_i_new - M_i[1, 2]) > ECC_MAX_DRIFT
            ):
                print(
                    f"[Stitch]   ECC frame {i}: correction clamped; keeping BA result."
                )
                refined.append(M_i)
                continue

            M_out = M_i.copy()
            M_out[0, 2] = tx_i_new
            M_out[1, 2] = ty_i_new
            refined.append(M_out)
            print(
                f"[Stitch]   ECC frame {i}: refined global dx={M_out[0, 2]:.2f} dy={M_out[1, 2]:.2f}"
            )

        except Exception as e:
            print(f"[Stitch]   ECC frame {i} failed ({e}); keeping BA result.")
            refined.append(M_i)

    return refined


__all__ = ["_ecc_refine"]
