import json
import os

from backend.controllers.backend_dispatch import _collect_image_paths, _run_single_stitch
from gui.src.helpers.gc_safe import gc_disabled_run
from PySide6.QtCore import QThread, Signal


class BatchStitchWorker(QThread):
    """
    GUI counterpart to the CLI's `stitch --batch-dir` mode (roadmap §4.1,
    Option A). Reuses the same `_collect_image_paths`/`_run_single_stitch`
    helpers the CLI batch path already validated, so both entry points stay
    behaviorally identical rather than diverging into two implementations of
    "run one sequence." Runs the plain (non-HITL) `AnimeStitchPipeline` path
    per subdirectory -- unattended batch runs don't pause for interactive
    review, unlike the single-sequence StitchTab/StitchWorker flow.

    Scans `batch_dir` for immediate subdirectories (each one a distinct
    frame group), stitches each into `{batch_dir}/{name}/{name}{suffix}.png`,
    and persists `.stitch_progress.json` in the same format the CLI writes
    so `resume=True` here can pick up a batch interrupted from either entry
    point (or vice versa).
    """

    sig_item_started = Signal(str, int, int)  # (seq_name, index, total)
    sig_item_finished = Signal(str, str)  # (seq_name, status: done/skipped/failed)
    sig_batch_finished = Signal(int, int, int)  # (done, skipped, failed)
    sig_log = Signal(str)

    def __init__(
        self,
        batch_dir: str,
        renderer: str = "median",
        output_suffix: str = "_stitched",
        resume: bool = False,
    ):
        super().__init__()
        self.batch_dir = batch_dir
        self.renderer = renderer
        self.output_suffix = output_suffix
        self.resume = resume
        self._should_stop = False

    def cancel(self):
        """Request cancellation. Checked between items, not mid-sequence --
        a single AnimeStitchPipeline.run() call isn't interruptible mid-stage
        (matches the project's existing §2.7B cancellation granularity)."""
        self._should_stop = True

    @gc_disabled_run
    def run(self):  # noqa: C901
        progress_path = os.path.join(self.batch_dir, ".stitch_progress.json")

        progress: dict = {}
        if self.resume and os.path.isfile(progress_path):
            try:
                with open(progress_path) as f:
                    progress = json.load(f)
            except Exception:
                progress = {}

        subdirs = sorted(
            d.path
            for d in os.scandir(self.batch_dir)
            if d.is_dir() and not d.name.startswith(".")
        )
        if not subdirs:
            self.sig_log.emit(f"No sub-directories found in '{self.batch_dir}'.")
            self.sig_batch_finished.emit(0, 0, 0)
            return

        total = len(subdirs)
        done = skipped = failed = 0

        for i, seq_dir in enumerate(subdirs, 1):
            if self._should_stop:
                self.sig_log.emit(f"Batch cancelled after {i - 1}/{total} sequence(s).")
                break

            seq_name = os.path.basename(seq_dir)
            out_path = os.path.join(seq_dir, f"{seq_name}{self.output_suffix}.png")
            self.sig_item_started.emit(seq_name, i, total)

            if self.resume and (
                os.path.isfile(out_path) or progress.get(seq_name) == "done"
            ):
                skipped += 1
                self.sig_item_finished.emit(seq_name, "skipped")
                continue

            image_paths = _collect_image_paths(seq_dir)
            if len(image_paths) < 2:
                progress[seq_name] = "skipped"
                failed += 1
                self.sig_item_finished.emit(seq_name, "skipped")
                self._persist_progress(progress_path, progress)
                continue

            success = _run_single_stitch(image_paths, out_path, self.renderer)
            if success:
                progress[seq_name] = "done"
                done += 1
                self.sig_item_finished.emit(seq_name, "done")
            else:
                progress[seq_name] = "failed"
                failed += 1
                self.sig_item_finished.emit(seq_name, "failed")

            self._persist_progress(progress_path, progress)

        self.sig_batch_finished.emit(done, skipped, failed)

    def _persist_progress(self, progress_path: str, progress: dict) -> None:
        try:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception:
            pass
