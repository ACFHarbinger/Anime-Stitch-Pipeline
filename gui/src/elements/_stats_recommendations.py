"""Statistics sub-tab: scenario-based stitching recommendation report.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change. Kept
as a single method: it's one continuous, static, non-``self`` computation
building up an ``obs``/``best_items``/``balanced_items``/``fast_items`` HTML
report -- there's no cohesive UI or state-owning group to split it against,
and its ``# noqa: C901`` was already present in the original file (this is
pre-existing complexity, not introduced by this split).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox


class _StatsRecommendationsMixin:
    # ------------------------------------------------------------------
    @staticmethod
    def _stats_build_recommendations(  # noqa: C901
        ind: list[dict],
        pw: list[dict],
        max_orb: int,  # noqa: ARG004
    ) -> str:
        """
        Analyse per-image and pairwise metrics and produce scenario-based
        stitching recommendations as an HTML string for display in QTextEdit.
        """
        n = len(ind)
        if n == 0:
            return "<p style='color:#888'>No data to analyse.</p>"

        # ── Derived aggregates ─────────────────────────────────────────
        sharpnesses = [float(r.get("sharpness", 0)) for r in ind]
        noises = [float(r.get("noise", 0)) for r in ind]
        brightnesses = [float(r.get("brightness", 0)) for r in ind]
        saturations = [float(r.get("saturation", 0)) for r in ind]
        widths = [int(r.get("width", 0)) for r in ind]
        heights = [int(r.get("height", 0)) for r in ind]
        file_sizes_kb = [float(r.get("file_size_kb", 0)) for r in ind]

        avg_sharp = float(np.mean(sharpnesses))
        std_sharp = float(np.std(sharpnesses))
        avg_noise = float(np.mean(noises))
        avg_bright = float(np.mean(brightnesses))
        bright_range = float(np.max(brightnesses) - np.min(brightnesses))
        avg_sat = float(np.mean(saturations))
        max_w = max(widths) if widths else 0
        max_h = max(heights) if heights else 0
        res_uniform = len(set(zip(widths, heights, strict=False))) == 1
        total_mb = sum(file_sizes_kb) / 1024

        # Pairwise aggregates — use consecutive pairs only for summary metrics
        # so the K-window extended pairs don't skew the quality verdict.
        _consec_pw = [r for r in pw if r.get("consecutive", True)] if pw else []
        _consec_pw = _consec_pw or (pw or [])  # fallback if flag absent
        scores = [float(r.get("_score", 0)) for r in _consec_pw]
        ssims = [float(r.get("ssim", 0)) for r in _consec_pw]
        orb_vals = [int(r.get("orb_inliers", 0)) for r in _consec_pw]

        avg_score = float(np.mean(scores)) if scores else 0.0
        avg_ssim = float(np.mean(ssims)) if ssims else 0.0
        avg_orb = float(np.mean(orb_vals)) if orb_vals else 0.0

        consec_pairs = [r for r in pw if r.get("consecutive", True)]
        consec_scores = [float(r.get("_score", 0)) for r in consec_pairs]
        weak_pairs = [consec_pairs[i] for i, s in enumerate(consec_scores) if s < 0.35]
        dup_pairs = [consec_pairs[i] for i, s in enumerate(consec_scores) if s > 0.92]
        n_weak = len(weak_pairs)
        n_dup = len(dup_pairs)

        # For each weak consecutive pair, check if any extended-window pair
        # involving either frame scores significantly better.
        extended_pairs = [r for r in pw if not r.get("consecutive", True)]
        better_matches: list[
            dict
        ] = []  # {weak_a, weak_b, best_a, best_b, score_orig, score_best}
        for wp in weak_pairs:
            orig_score = float(wp.get("_score", 0))
            ia, ib = wp["idx_a"], wp["idx_b"]
            candidates = [
                r
                for r in extended_pairs
                if r["idx_a"] == ia
                or r["idx_b"] == ib
                or r["idx_a"] == ib
                or r["idx_b"] == ia
            ]
            if not candidates:
                continue
            best_ext = max(candidates, key=lambda r: float(r.get("_score", 0)))
            best_score = float(best_ext.get("_score", 0))
            if best_score > orig_score + 0.10:  # only report meaningful improvement
                better_matches.append(
                    {
                        "weak_a": wp["name_a"],
                        "weak_b": wp["name_b"],
                        "best_a": best_ext["name_a"],
                        "best_b": best_ext["name_b"],
                        "score_orig": orig_score,
                        "score_best": best_score,
                        "gap": abs(best_ext["idx_b"] - best_ext["idx_a"]),
                    }
                )

        # ── Observations (shared facts) ────────────────────────────────
        obs: list[str] = []

        # ── Sharpness — relative to the batch, not absolute ──────────────
        # Digital anime / cel-shading has low Laplacian variance by design
        # (flat colour fills, no film grain).  We use the batch's own
        # distribution so the verdict is always content-appropriate.
        sharp_cv = (
            std_sharp / avg_sharp if avg_sharp > 0 else 0.0
        )  # coefficient of variation

        if sharp_cv < 0.25:
            # Frames are very uniform in sharpness — describe the batch level
            # relative to typical photographic expectations, but without
            # labelling anime as "blurry" just because its absolute values
            # are low (which is normal for cel-shaded content).
            obs.append(
                f"✔ Sharpness is <b>consistent</b> across all frames "
                f"(avg Laplacian variance {avg_sharp:.0f}, CV {sharp_cv:.2f}).  "
                "This is normal for digital anime / cel-shaded content — "
                "flat colour fills inherently produce low Laplacian values."
            )
        else:
            obs.append(
                f"○ Sharpness varies across frames "
                f"(avg {avg_sharp:.0f} ± {std_sharp:.0f}).  "
                "Some frames may be motion-blurred or out-of-focus."
            )

        # Flag genuine outliers: frames more than 2σ below the mean AND
        # at least 30% softer than the mean (prevents false positives when
        # the whole batch is a tight cluster, e.g. uniform anime content).
        # Never recommend removing more than n-2 frames (must keep ≥ 2).
        if n > 2 and std_sharp > 0:
            two_sigma_thresh = avg_sharp - 2.0 * std_sharp
            thirty_pct_thresh = avg_sharp * 0.70
            outlier_thresh = max(two_sigma_thresh, thirty_pct_thresh)
            blurry_outliers = [
                ind[i]["name"] for i, v in enumerate(sharpnesses) if v < outlier_thresh
            ]
            # Hard cap: never suggest removing so many frames stitching breaks
            max_removable = max(0, n - 2)
            blurry_outliers = blurry_outliers[:max_removable]
            if blurry_outliers:
                obs.append(
                    f"⚠ {len(blurry_outliers)} frame(s) are significantly softer "
                    f"than the rest of the batch (>2σ below average): "
                    f"<i>{', '.join(blurry_outliers[:4])}"
                    f"{'…' if len(blurry_outliers) > 4 else ''}</i>.  "
                    "These may be motion-blurred transitions — consider removing them."
                )

        # Noise
        if avg_noise > 15:
            obs.append(
                f"⚠ High average noise ({avg_noise:.1f}).  "
                "Pre-denoising (e.g. Gaussian or bilateral filter) is recommended."
            )
        elif avg_noise > 5:
            obs.append(
                f"○ Mild noise ({avg_noise:.1f}).  "
                "Optional light denoising before high-quality stitching."
            )

        # Brightness / exposure
        if bright_range > 60:
            obs.append(
                f"⚠ <b>Large brightness variation</b> across frames "
                f"(range {bright_range:.0f}).  "
                "Exposure normalisation is strongly recommended before stitching "
                "to avoid seam-line banding."
            )
        elif bright_range > 30:
            obs.append(
                f"○ Moderate brightness spread ({bright_range:.0f}).  "
                "Histogram matching between adjacent pairs will improve seam quality."
            )

        if avg_bright < 60:
            obs.append(
                "⚠ Frames are <b>underexposed</b>.  "
                "Gamma correction or CLAHE may improve feature detection."
            )
        elif avg_bright > 200:
            obs.append(
                "⚠ Frames are <b>overexposed</b>.  "
                "Tone-map or reduce brightness to recover detail before stitching."
            )

        # Resolution uniformity
        if not res_uniform:
            obs.append(
                "⚠ Frames have <b>inconsistent resolutions</b>.  "
                "Resize all frames to a common resolution before stitching "
                "to avoid projection errors."
            )
        else:
            obs.append(f"✔ All frames share the same resolution ({max_w}×{max_h}).")

        # Saturation / colour
        if avg_sat < 30:
            obs.append(
                "○ Frames appear mostly desaturated.  "
                "Histogram correlation will be less discriminative; "
                "rely on structural matchers (LoFTR/ORB)."
            )

        # Pairwise quality
        if pw:
            if avg_score >= 0.65:
                obs.append(
                    f"✔ Pair match quality is <b>strong</b> "
                    f"(avg stitch score {avg_score:.2f}).  "
                    "Direct stitching is likely to succeed."
                )
            elif avg_score >= 0.40:
                obs.append(
                    f"○ Pair match quality is <b>acceptable</b> "
                    f"(avg stitch score {avg_score:.2f}).  "
                    "Light pre-processing will improve results."
                )
            else:
                obs.append(
                    f"⚠ Pair match quality is <b>weak</b> "
                    f"(avg stitch score {avg_score:.2f}).  "
                    "Significant pre-processing or manual alignment may be required."
                )

            if n_weak:
                names_w = [f"{p['name_a']}↔{p['name_b']}" for p in weak_pairs[:3]]
                obs.append(
                    f"⚠ {n_weak} weak consecutive pair(s) detected: "
                    f"<i>{', '.join(names_w)}"
                    f"{'…' if n_weak > 3 else ''}</i>.  "
                    "These transitions may produce visible seams or misalignments."
                )

            if better_matches:
                for bm in better_matches[:5]:
                    obs.append(
                        f"💡 Weak pair <i>{bm['weak_a']}↔{bm['weak_b']}</i> "
                        f"(score {bm['score_orig']:.3f}) has a better non-adjacent match: "
                        f"<b>{bm['best_a']}↔{bm['best_b']}</b> "
                        f"(score <b>{bm['score_best']:.3f}</b>, gap {bm['gap']} frame(s)).  "
                        "Consider replacing the weak pair with this one or skipping intermediate frames."
                    )
                if len(better_matches) > 5:
                    obs.append(
                        f"… and {len(better_matches) - 5} more weak pair(s) with better "
                        "non-adjacent alternatives (see table)."
                    )

            if n_dup:
                names_d = [f"{p['name_a']}↔{p['name_b']}" for p in dup_pairs[:3]]
                obs.append(
                    f"⚠ {n_dup} near-duplicate pair(s) detected: "
                    f"<i>{', '.join(names_d)}</i>.  "
                    "Redundant frames add compute cost without improving coverage — "
                    "consider removing one from each duplicate pair."
                )

        # Memory / size
        if total_mb > 500:
            obs.append(
                f"⚠ Total uncompressed data is large (~{total_mb:.0f} MB).  "
                "Downscale frames before stitching unless maximum output resolution "
                "is required."
            )

        # ── Scenario recommendations ───────────────────────────────────
        def _section(title: str, color: str, items: list[str]) -> str:
            bullets = "".join(f"<li>{it}</li>" for it in items)
            return (
                f"<div style='margin-top:10px;'>"
                f"<span style='background:{color};color:#fff;"
                f"font-weight:bold;padding:2px 8px;border-radius:3px;'>"
                f"{title}</span>"
                f"<ul style='margin:4px 0 0 16px;padding:0;'>{bullets}</ul>"
                f"</div>"
            )

        best_items: list[str] = []
        balanced_items: list[str] = []
        fast_items: list[str] = []

        # Resolution advice
        if max_w > 0:
            best_items.append(
                f"Use <b>full native resolution</b> ({max_w}×{max_h}) throughout."
            )
            half_w, half_h = max_w // 2, max_h // 2
            balanced_items.append(
                f"Use <b>½ resolution</b> ({half_w}×{half_h}) for stitching, "
                f"then upscale the result."
            )
            q_w, q_h = max_w // 4, max_h // 4
            fast_items.append(
                f"Downscale to <b>¼ resolution</b> ({q_w}×{q_h}) for a rapid prototype."
            )

        # Pre-processing
        if avg_noise > 5:
            best_items.append(
                "Apply <b>bilateral denoising</b> (preserve edges) before matching."
            )
            balanced_items.append(
                "Apply a <b>light Gaussian blur</b> (σ=0.8) to suppress noise."
            )
            fast_items.append("Skip denoising — accept minor artefacts.")

        if bright_range > 30:
            best_items.append(
                "Run <b>per-frame exposure normalisation</b> (CLAHE or histogram matching) "
                "before stitching, then gain-compensate during blending."
            )
            balanced_items.append(
                "Apply <b>histogram matching</b> between consecutive frame pairs."
            )
            fast_items.append(
                "Use a simple <b>global mean-brightness normalisation</b>."
            )

        # Feature matching — decide based on whether outliers exist, not
        # on absolute sharpness (which is content-type dependent).
        has_blur_outliers = (
            n > 2
            and std_sharp > 0
            and (avg_sharp - 2.0 * std_sharp) > float(np.percentile(sharpnesses, 10))
        )
        if has_blur_outliers and sharp_cv >= 0.25:
            best_items.append(
                "Use <b>LoFTR</b> (transformer-based) matching — it handles "
                "motion-blurred frames far better than ORB."
            )
            balanced_items.append(
                "Use <b>LoFTR</b> with a reduced tile size to save memory."
            )
            fast_items.append(
                "Use <b>ORB + BFMatcher</b> — fast, but expect fewer inliers "
                "on the blurry outlier frames."
            )
        else:
            best_items.append("Use <b>LoFTR</b> for maximum correspondence quality.")
            balanced_items.append(
                "Use <b>LoFTR</b> or SIFT — sharpness is consistent across "
                "the batch, both will work well."
            )
            fast_items.append(
                "Use <b>ORB + BFMatcher</b> — sharpness is uniform, "
                "ORB inlier counts will be reliable."
            )

        # Homography / warp
        if avg_orb > 80 or (avg_ssim > 0.7 and avg_orb > 40):
            best_items.append(
                "Estimate a full <b>homography</b> (8-DOF perspective) per pair — "
                "feature density supports it."
            )
            balanced_items.append(
                "Use an <b>affine</b> warp (6-DOF) for speed while retaining most accuracy."
            )
        else:
            best_items.append(
                "Use <b>affine + RANSAC</b> warp — feature count may be too low "
                "for reliable homography estimation."
            )
            balanced_items.append(
                "Use <b>affine</b> warp with a relaxed RANSAC threshold (8–10 px)."
            )
        fast_items.append(
            "Use a <b>translation-only</b> alignment (fast, sufficient for roughly "
            "pre-aligned frames)."
        )

        # Blending
        best_items.append(
            "Use <b>multi-band blending</b> (Laplacian pyramid, 5–6 levels) "
            "for seamless transitions."
        )
        if n_weak:
            best_items.append(
                f"Manually verify / override affine for the {n_weak} weak pair(s) "
                "before blending."
            )
        balanced_items.append(
            "Use <b>linear feathering</b> (alpha blend) across a 20–40 px overlap zone."
        )
        fast_items.append(
            "Use a <b>hard cut</b> at the midpoint of the overlap — no blending."
        )

        # Output format
        best_items.append(
            "Save as <b>PNG</b> (lossless) or 16-bit TIFF for archival quality."
        )
        balanced_items.append(
            "Save as <b>JPEG quality 92–95</b> for a good size/quality trade-off."
        )
        fast_items.append(
            "Save as <b>JPEG quality 80</b> or WebP for minimal disk footprint."
        )

        # Frame count / compute note
        if n >= 8:
            best_items.append(
                f"With {n} frames, consider enabling the <b>graph-based stitcher</b> "
                "(Graph tab) to find the globally optimal stitch order."
            )
            balanced_items.append(
                f"With {n} frames, use <b>Auto-Order</b> to optimise frame sequence "
                "before stitching."
            )
            fast_items.append(
                f"Limit the fast prototype to the <b>highest-scoring {min(n, 4)} frames</b> "
                "to keep runtime under a few seconds."
            )

        if n_dup:
            best_items.append(
                f"Remove {n_dup} near-duplicate frame(s) to avoid redundant blending "
                "and output ghosting."
            )
            balanced_items.append(
                f"Remove {n_dup} near-duplicate frame(s) before stitching."
            )
            fast_items.append("Skip duplicate frames entirely.")

        # ── Assemble HTML ──────────────────────────────────────────────
        obs_html = "".join(f"<li style='margin-bottom:2px;'>{o}</li>" for o in obs)

        html = (
            "<html><body style='font-family:sans-serif;font-size:11px;"
            "color:#d4d4d4;background:#1e1f22;'>"
            "<p style='font-weight:bold;font-size:12px;margin:0 0 4px;'>"
            "Observations</p>"
            f"<ul style='margin:0 0 6px 16px;padding:0;'>{obs_html}</ul>"
            "<hr style='border:1px solid #4f545c;margin:8px 0;'/>"
            "<p style='font-weight:bold;font-size:12px;margin:0 0 2px;'>"
            "Scenario Recommendations</p>"
        )
        html += _section("Best Quality", "#1976D2", best_items)
        html += _section("Balanced", "#388E3C", balanced_items)
        html += _section("Fast / Prototype", "#F57C00", fast_items)
        html += "</body></html>"
        return html

    @Slot(str)
    def _stats_on_error(self, msg: str):
        self._stats_progress.hide()
        self._stats_status.setText("Error.")
        self._stats_run_btn.setEnabled(True)
        self._stats_worker = None
        QMessageBox.critical(self, "Statistics Error", msg)


__all__ = ["_StatsRecommendationsMixin"]
