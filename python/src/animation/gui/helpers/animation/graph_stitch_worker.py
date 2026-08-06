from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import QThread, Signal

from .stitch_worker import (
    _TOTAL_STAGES,
    _build_pipeline_kwargs,
    _ProgressPipeline,
)


class GraphStitchWorker(QThread):
    """
    Executes a DAG of stitch operations in topological order.

    Each plan step is a dict:
        {
          "id":     str,         # unique step identifier (referenced by later steps)
          "name":   str,         # display name
          "inputs": list[str],   # image paths or step IDs from earlier steps
          "output": str,         # output file path
        }

    Steps are executed sequentially in list order. After each step its output
    path is stored under its ID so later steps can reference it by ID.
    """

    sig_step = Signal(int, int, str)  # (current_step, total_steps, step_name)
    sig_stage = Signal(int, int, str)  # (stage, total_stages, label) within step
    sig_log = Signal(str)
    sig_finished = Signal(list)  # list of output paths
    sig_error = Signal(str)

    def __init__(self, plan: List[Dict], pipeline_config: dict):
        super().__init__()
        self._plan = plan
        self._cfg = pipeline_config
        self._cancel_flag: list = [False]

    def cancel(self):
        self._cancel_flag[0] = True

    def run(self):
        cfg = self._cfg
        step_outputs: Dict[str, str] = {}
        output_paths: List[str] = []
        total = len(self._plan)

        for idx, step in enumerate(self._plan):
            if self._cancel_flag[0]:
                self.sig_error.emit("Cancelled.")
                return

            step_id = step.get("id", f"step_{idx}")
            step_name = step.get("name", f"Step {idx + 1}")
            self.sig_step.emit(idx + 1, total, step_name)
            self.sig_log.emit(f"\n=== {step_name} ===")

            # Resolve inputs: may be file paths or IDs of earlier steps
            resolved: List[str] = []
            for inp in step.get("inputs", []):
                resolved.append(step_outputs.get(inp, inp))

            if len(resolved) < 2:
                self.sig_error.emit(
                    f"Step '{step_name}' needs ≥ 2 inputs; got {len(resolved)}."
                )
                return

            out_path = step.get("output", "")
            if not out_path:
                self.sig_error.emit(f"Step '{step_name}' has no output path set.")
                return

            def _progress_cb(stage_idx: int, label: str):
                self.sig_stage.emit(stage_idx, _TOTAL_STAGES, label)

            def _log_cb(msg: str):
                self.sig_log.emit(msg)

            try:
                pipeline = _ProgressPipeline(
                    progress_cb=_progress_cb,
                    log_cb=_log_cb,
                    cancel_flag=self._cancel_flag,
                    mfsr_n_dct_iter=cfg.get("mfsr_n_dct_iter", 20),
                    mfsr_use_prior=cfg.get("mfsr_use_prior", True),
                    mfsr_use_diffusion=cfg.get("mfsr_use_diffusion", False),
                    **_build_pipeline_kwargs(cfg),
                )
                pipeline.run(resolved, out_path)
                step_outputs[step_id] = out_path
                output_paths.append(out_path)
                self.sig_log.emit(f"[Graph] '{step_name}' → '{out_path}'")
            except InterruptedError:
                self.sig_error.emit("Cancelled.")
                return
            except Exception as e:
                self.sig_error.emit(f"Step '{step_name}' failed: {e}")
                return

        self.sig_finished.emit(output_paths)
