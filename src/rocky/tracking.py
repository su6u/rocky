"""Explicit W&B sessions with scalar-only logs and local recovery records."""

from __future__ import annotations

import importlib.util
import json
import math
import time
import uuid
from pathlib import Path


def check_tracking(mode: str, project: str) -> None:
    if mode not in {"offline", "online", "disabled"} or not project.strip():
        raise ValueError("choose offline, online, or disabled tracking and a nonempty project")
    if mode != "disabled" and importlib.util.find_spec("wandb") is None:
        raise ValueError('W&B is missing; install rocky[tracking] or use --wandb-mode disabled')


class RunTracking:
    def __init__(self, output: Path, *, mode: str, project: str, entity: str | None,
                 identity: dict, diagnostics: dict, hourly_usd: float):
        self.output, self.hourly_usd = output, hourly_usd
        self.started = time.monotonic()
        self.run = None
        self.metadata = {"mode": mode}
        if mode == "disabled":
            return
        import wandb

        # Each execution attempt has a separate run, including checkpoint resumes.
        # This preserves rolled-back steps and works offline, where resume is unsupported.
        tracking_file = output / "tracking.json"
        attempts = json.loads(tracking_file.read_text()) if tracking_file.exists() else []
        attempt_id = uuid.uuid4().hex[:16]
        group = attempts[0]["group"] if attempts else "rocky-" + attempt_id
        run = wandb.init(project=project, entity=entity, mode=mode, id=attempt_id,
                         name=f"{output.name}-attempt-{len(attempts) + 1}", group=group,
                         job_type=identity["objective"], dir=str(output.resolve()),
                         config={"identity": identity, "diagnostics": diagnostics,
                                 "previous_attempt": attempts[-1]["id"] if attempts else None},
                         save_code=False, settings=wandb.Settings(disable_git=True, console="off"))
        self.run = run
        try:
            run.define_metric("trainer/step")
            run.define_metric("*", step_metric="trainer/step")
            self.metadata = {"mode": mode, "id": attempt_id, "group": group,
                             "project": project, "entity": entity, "url": run.url if mode == "online" else None}
            attempts.append(self.metadata)
            tracking_file.write_text(json.dumps(attempts, indent=2) + "\n")
            groups = diagnostics["dataset"]["groups"]
            columns = ["split", "source", "conversations", "replies", "target_words"]
            run.log({"data/composition": wandb.Table(columns=columns,
                     data=[[row[key] for key in columns] for row in groups]), "trainer/step": 0})
        except BaseException:
            run.finish(exit_code=1)
            raise

    def log(self, step: int, metrics: dict) -> None:
        values = {key: float(value) for key, value in metrics.items()
                  if isinstance(value, (int, float)) and math.isfinite(value)}
        seconds = time.monotonic() - self.started
        values.update({"trainer/step": step, "system/elapsed_seconds": seconds,
                       "system/estimated_gpu_cost_usd": seconds * self.hourly_usd / 3600})
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"attempt": self.metadata.get("id"), **values}) + "\n")
        if self.run:
            self.run.log(values)

    def finish(self, record: dict, exit_code: int = 0) -> None:
        if self.run:
            try:
                self.run.summary.update({key: record[key] for key in (
                    "status", "persona_accepted", "elapsed_seconds", "estimated_gpu_cost_usd",
                    "peak_allocated_gib", "peak_reserved_gib", "final_metrics", "error_type") if key in record})
            finally:
                self.run.finish(exit_code=exit_code)


def tracking_callback(tracker: RunTracking):
    import torch
    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {}
            for key, value in (logs or {}).items():
                name = "eval/" + key[5:] if key.startswith("eval_") else "train/" + key
                values[name] = value
            if torch.cuda.is_available():
                values.update({"system/allocated_gib": torch.cuda.memory_allocated() / 2**30,
                               "system/reserved_gib": torch.cuda.memory_reserved() / 2**30,
                               "system/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
            tracker.log(state.global_step, values)

    return MetricsCallback()
