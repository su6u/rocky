"""verify fail-closed checkpoint selection through one interface"""

import json
from pathlib import Path

import pytest

from rocky_training.checkpoint_select import (
    CheckpointCandidate,
    CheckpointSelectError,
    composite_gate_score,
    select_best_checkpoint,
    select_best_from_eval_dir,
)


def _gates(**overrides: object) -> dict[str, object]:
    base = {
        "responseSchemaValidRate": 1.0,
        "responseSingleObjectRate": 1.0,
        "rockyPersonaRate": 1.0,
        "bookFactContradictionRate": 0.0,
        "promptInjectionFailRate": 0.0,
        "failures": [],
        "passed": True,
    }
    base.update(overrides)
    return base


def test_composite_gate_score_prefers_higher_persona() -> None:
    weak = composite_gate_score(_gates(rockyPersonaRate=0.5))
    strong = composite_gate_score(_gates(rockyPersonaRate=1.0))
    assert strong > weak


def test_select_best_by_eval_loss() -> None:
    best = select_best_checkpoint(
        [
            CheckpointCandidate("a", Path("a"), eval_loss=0.4, gate_summary=None),
            CheckpointCandidate("b", Path("b"), eval_loss=0.2, gate_summary=None),
        ],
        metric="eval_loss",
    )
    assert best.checkpoint_id == "b"


def test_select_best_by_composite_gates() -> None:
    best = select_best_checkpoint(
        [
            CheckpointCandidate(
                "ineligible",
                Path("ineligible"),
                eval_loss=0.1,
                gate_summary=_gates(
                    rockyPersonaRate=1.0,
                    passed=False,
                    failures=["relationship trajectory failed"],
                ),
            ),
            CheckpointCandidate(
                "high", Path("high"), eval_loss=0.5, gate_summary=_gates(rockyPersonaRate=0.95)
            ),
        ],
        metric="composite_gates",
    )
    assert best.checkpoint_id == "high"


def test_select_best_from_eval_dir(tmp_path: Path) -> None:
    (tmp_path / "ckpta.json").write_text(
        json.dumps(
            {
                "label": "cka",
                "adapterDir": str(tmp_path / "adapter-a"),
                "evalLoss": 0.3,
                "gateSummary": _gates(
                    rockyPersonaRate=1.0,
                    passed=False,
                    failures=["callback near-miss gate failed"],
                ),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ckptb.json").write_text(
        json.dumps(
            {
                "label": "ckb",
                "adapterDir": str(tmp_path / "adapter-b"),
                "evalLoss": 0.9,
                "gateSummary": _gates(rockyPersonaRate=0.99),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "sidecar.results.json").write_text("[]\n", encoding="utf-8")

    report = select_best_from_eval_dir(tmp_path, metric="composite_gates")
    assert report["best"]["checkpointId"] == "ckb"
    assert len(report["ranked"]) == 1


def test_select_requires_gate_summary_for_composite() -> None:
    with pytest.raises(CheckpointSelectError, match="missing gateSummary"):
        select_best_checkpoint(
            [CheckpointCandidate("x", Path("x"), eval_loss=0.1, gate_summary=None)],
            metric="composite_gates",
        )

    with pytest.raises(CheckpointSelectError, match="no checkpoint candidate passed"):
        select_best_checkpoint(
            [
                CheckpointCandidate(
                    "x",
                    Path("x"),
                    eval_loss=0.1,
                    gate_summary=_gates(passed=False, failures=["persona failed"]),
                )
            ],
            metric="composite_gates",
        )
