"""select checkpoints only after every release gate passes"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CheckpointSelectError(Exception):
    pass


@dataclass(frozen=True)
class CheckpointCandidate:
    checkpoint_id: str
    path: Path
    eval_loss: float | None
    gate_summary: dict[str, Any] | None


def composite_gate_score(gate_summary: dict[str, Any]) -> float:
    """Higher is better. Blends release-gate rates into one score in [0, 1]."""
    higher = [
        float(gate_summary.get("responseSchemaValidRate", 0.0)),
        float(gate_summary.get("responseSingleObjectRate", 0.0)),
        float(gate_summary.get("rockyPersonaRate", 0.0)),
    ]
    for key in (
        "personaRubricPassRate",
        "humorExpectedPassRate",
        "humorAvoidPassRate",
    ):
        value = gate_summary.get(key)
        if isinstance(value, (int, float)):
            higher.append(float(value))
    lower = (
        float(gate_summary.get("bookFactContradictionRate", 1.0)),
        float(gate_summary.get("promptInjectionFailRate", 1.0)),
    )
    higher_mean = sum(higher) / len(higher)
    lower_mean = sum(1.0 - rate for rate in lower) / len(lower)
    return (higher_mean + lower_mean) / 2.0


def score_candidate(candidate: CheckpointCandidate, *, metric: str) -> float:
    if metric == "eval_loss":
        if candidate.eval_loss is None:
            raise CheckpointSelectError(f"{candidate.checkpoint_id}: missing eval_loss")
        return -float(candidate.eval_loss)
    if metric == "composite_gates":
        if candidate.gate_summary is None:
            raise CheckpointSelectError(f"{candidate.checkpoint_id}: missing gateSummary")
        return composite_gate_score(candidate.gate_summary)
    raise CheckpointSelectError(f"unsupported checkpoint metric: {metric}")


def select_best_checkpoint(
    candidates: list[CheckpointCandidate],
    *,
    metric: str = "composite_gates",
) -> CheckpointCandidate:
    if not candidates:
        raise CheckpointSelectError("no checkpoint candidates")
    if metric == "composite_gates":
        for candidate in candidates:
            if candidate.gate_summary is None:
                raise CheckpointSelectError(f"{candidate.checkpoint_id}: missing gateSummary")
        candidates = [
            candidate
            for candidate in candidates
            if candidate.gate_summary is not None
            and candidate.gate_summary.get("passed") is True
            and not candidate.gate_summary.get("failures")
        ]
        if not candidates:
            raise CheckpointSelectError("no checkpoint candidate passed every release gate")
    ranked = sorted(
        candidates,
        key=lambda candidate: score_candidate(candidate, metric=metric),
        reverse=True,
    )
    return ranked[0]


def load_checkpoint_candidate(eval_payload_path: Path) -> CheckpointCandidate:
    payload = json.loads(eval_payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CheckpointSelectError(f"{eval_payload_path}: payload must be an object")
    checkpoint_id = str(payload.get("label") or eval_payload_path.stem)
    gate_summary = payload.get("gateSummary")
    if gate_summary is not None and not isinstance(gate_summary, dict):
        raise CheckpointSelectError(f"{eval_payload_path}: gateSummary must be an object")
    eval_loss_raw = payload.get("evalLoss")
    eval_loss = float(eval_loss_raw) if isinstance(eval_loss_raw, (int, float)) else None
    adapter_dir = payload.get("adapterDir") or payload.get("checkpointPath")
    path = Path(str(adapter_dir)) if adapter_dir else eval_payload_path
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        path=path,
        eval_loss=eval_loss,
        gate_summary=gate_summary if isinstance(gate_summary, dict) else None,
    )


def select_best_from_eval_dir(
    eval_dir: Path,
    *,
    metric: str = "composite_gates",
) -> dict[str, Any]:
    paths = sorted(eval_dir.glob("*.json"))
    # Prefer full eval payloads over .results.json sidecars
    paths = [path for path in paths if not path.name.endswith(".results.json")]
    if not paths:
        raise CheckpointSelectError(f"no eval json files in {eval_dir}")
    candidates = [load_checkpoint_candidate(path) for path in paths]
    best = select_best_checkpoint(candidates, metric=metric)
    eligible = (
        [
            candidate
            for candidate in candidates
            if candidate.gate_summary is not None
            and candidate.gate_summary.get("passed") is True
            and not candidate.gate_summary.get("failures")
        ]
        if metric == "composite_gates"
        else candidates
    )
    ranked = [
        {
            "checkpointId": candidate.checkpoint_id,
            "path": str(candidate.path),
            "score": score_candidate(candidate, metric=metric),
            "evalLoss": candidate.eval_loss,
        }
        for candidate in sorted(
            eligible,
            key=lambda item: score_candidate(item, metric=metric),
            reverse=True,
        )
    ]
    return {
        "kind": "checkpoint-select",
        "metric": metric,
        "best": {
            "checkpointId": best.checkpoint_id,
            "path": str(best.path),
            "score": score_candidate(best, metric=metric),
        },
        "ranked": ranked,
    }
