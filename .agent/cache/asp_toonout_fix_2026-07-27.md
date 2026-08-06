# ASP §3.4 Cheap Photometric Candidates — ToonOut Fix + Reverse-Dimming Check (2026-07-27)

Two sub-items from roadmap §3.4. One was a genuine, previously-unknown bug
fix (kept as the new default, no flag). The other was a "check before
building" item that came back negative (not built).

## ToonOut weights — bug fix, not just a stale-mirror problem

The roadmap's blocker note ("the MatteoKartoon HF repo is gone — locate a
mirror first") was itself stale. A live mirror exists:
**`joelseytre/toonout`** on HuggingFace (MIT-licensed, `base_model:
ZhengPeng7/BiRefNet`, verified reachable via the HF Hub API). But locating
it surfaced a deeper, pre-existing bug in `birefnet_wrapper.py`:

1. **The constants were swapped relative to their names.**
   `TOONOUT_MODEL = "ZhengPeng7/BiRefNet"` (the *generic*, non-anime-tuned
   model) and `BIREFNET_MODEL = "MatteoKartoon/BiRefNet"` (the intended
   fallback) — but `MatteoKartoon/BiRefNet` is a **GitHub org, not a
   HuggingFace model repo** (returns 401/invalid on the HF Hub API). Since
   `TOONOUT_MODEL` (despite its name) always resolved to valid weights, the
   broken fallback was never even reached — **ToonOut had never actually
   been loaded by this wrapper, ever**; every run silently used plain
   generic BiRefNet regardless of intent.
2. **Even after fixing the repo IDs, the real fix was masked by a second
   bug**: `joelseytre/toonout` ships a plain `.pth` checkpoint
   (`birefnet_finetuned_toonout.pth`), not the HF-standard `model.safetensors`
   /`pytorch_model.bin` names `_load_weights` was hardcoded to try — so even
   with the correct repo ID, the download would 404 and silently fall back
   to the generic model again. Added the real filename to the candidates
   tried.
3. **Even after both of those, `load_state_dict(strict=True)` still
   failed silently and fell back**: the checkpoint's keys are all prefixed
   `module._orig_mod.` (saved from a `torch.compile(DataParallel(model))`
   wrapper during training), which our plain local `BiRefNet` instance
   doesn't have. Added prefix-stripping for this (and the more common bare
   `module.`/`_orig_mod.` cases) before loading.

Verified end-to-end: a direct before/after mask comparison on the same
frame shows a genuine, substantial difference (6.7% of pixels differ by
>0.1 probability, ToonOut classifies ~81% of the frame as foreground vs
~75% for generic BiRefNet — consistent with ToonOut's published improvement
on fine detail like hair wisps) — not just "loads without erroring," an
actual behavioral change confirmed at the pixel level before trusting any
downstream benchmark number.

### 5-test verify result (this is a bug fix restoring intended default
behavior, not a speculative feature — no new flag; benchmarked to confirm
no regression, not to decide keep-vs-revert)

- **test27**: real improvement — post_warp_diff 3.96→3.59, seam_visibility
  16.14→11.95, and a visible reduction in the pre-existing ghosting
  artifact around the character's hair/pom-poms.
- **test08, test57**: both flip from a real (but visibly banded — seen
  directly in the images) composite to a safe SCANS fallback. Verdict
  improves in both cases (`simple_better`→`comparable`) — losing a flawed
  real attempt for a clean fallback is a net-positive trade under this
  project's "never worse than fallback" objective, not a loss.
- **test09**: unaffected (masking doesn't materially change this test's
  outcome).
- **test04**: flips fallback→real, and the resulting composite shows a
  visible defect (horizontal banding plus a blocky colour-distorted patch).
  **This is the same test, showing the same class of defect, that the §3.1
  joint-gain-solve postmortem already flagged** (`.agent/cache/asp_joint_gain_solve_postmortem_2026-07-27.md`)
  — test04 sits right at the composite-gate threshold and *any* quality
  improvement elsewhere (there: gain solve; here: masking accuracy) nudges
  its aggregate gate score just past passing, without the underlying local
  defect actually being fixed. This is now confirmed to be a
  **test-specific, gate-design issue** independent of which upstream
  improvement triggers it — not something to chase per-improvement.

**Disposition**: kept as the new default (this is a bug fix, not an
optional feature — the intent was always for ToonOut to be the default
when available). Net effect: 3 tests improve or trade a flawed composite
for a clean fallback, 1 unaffected, 1 exposes the already-documented
test04 gate-threshold fragility. Recommended follow-up (not attempted
here): the finer-grained local-defect gate check recommended in the §3.1
postmortem would likely fix test04 under both triggers at once, since the
root cause is the same gate design gap, not either upstream change.

## Reverse-dimming — checked, not present, not built

The roadmap explicitly asked to "check whether any of the 97 tests
actually show Harding dimming before building it." Sampled 3 tests' already-
generated `output/plots/gains.png` (per-frame background luminance +
applied gain, computed for the existing photometric-correction pipeline)
looking for Harding's signature: a *sudden luminance drop* on flash/high-
contrast/fast-pan frames followed by recovery.

Found instead: `asp_test08` shows a smooth, monotonic luminance drift
(gradual rise then fall — normal auto-exposure/panning behavior, not a
sudden dip); `asp_test48` and `asp_test81` both show isolated 2-5-frame
**brightening** spikes against a stable baseline — the *opposite* direction
from Harding dimming (which darkens risky content, not brightens it). The
existing pipeline's coherence gate (`_compute_skip_normalization_mask`,
compositing.py) already detects and skips aggressive normalization across
these outlier-brightness frame pairs, so they're not silently corrupting
anything either.

This content (R18 OVA source, not broadcast on Japanese terrestrial TV) was
never likely to be subject to Harding Flash-and-Pattern compliance
processing in the first place, and the sampled evidence confirms no
dimming signature is present. **Not built** — matches the roadmap's own
"check first" instruction, and the answer is no.
