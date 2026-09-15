import json
from types import SimpleNamespace

import pytest

from rocky.analysis import near_overlap, token_diagnostics
from rocky.cli import build_parser
from rocky.tracking import RunTracking, check_tracking


def test_near_overlap_flags_paraphrase_with_shared_phrase_only_across_splits():
    text = "The collection instrument stopped recording halfway through the second measurement today"
    rows = [{"id": "a", "split": "train", "messages": [{"content": text}]},
            {"id": "b", "split": "validation", "messages": [{"content": text + " unfortunately"}]},
            {"id": "c", "split": "train", "messages": [{"content": text}]}]
    result = near_overlap(rows)
    assert {row["heldout_id"] for row in result} == {"b"}
    assert len(result) == 2


def test_token_diagnostics_include_both_preference_alternatives():
    chosen = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    rejected = {"input_ids": [1, 4], "labels": [-100, 4]}
    result = token_diagnostics([{"chosen": chosen, "rejected": rejected}])
    assert result["target_tokens"] == 3
    assert result["input_tokens"] == 5
    assert result["supervised_fraction"] == 0.6


def test_tracking_missing_dependency_is_actionable(monkeypatch):
    monkeypatch.setattr("rocky.tracking.importlib.util.find_spec", lambda name: None)
    check_tracking("disabled", "rocky")
    with pytest.raises(ValueError, match="install rocky"):
        check_tracking("offline", "rocky")
    args = build_parser().parse_args(["train-sft", "--preflight"])
    assert args.wandb_mode == "offline"


def test_offline_attempts_are_linked_without_unsupported_resume(monkeypatch, tmp_path):
    calls, runs = [], []

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.logs = []
            self.exit_code = None

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values):
            self.logs.append(values)

        def finish(self, exit_code=0):
            self.exit_code = exit_code

    def init(**kwargs):
        calls.append(kwargs)
        run = FakeRun()
        runs.append(run)
        return run

    import sys

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, Settings=lambda **kw: kw, Table=lambda **kw: kw))
    for attempt in range(2):
        tracker = RunTracking(tmp_path, mode="offline", project="rocky", entity=None,
                              identity={"objective": "sft"}, diagnostics={"dataset": {"groups": []}}, hourly_usd=1)
        tracker.log(5, {"train/loss": 1.2, "bad": float("nan"), "text": "not logged"})
        tracker.finish({"status": "completed", "persona_accepted": False})
    assert calls[0]["id"] != calls[1]["id"]
    assert calls[0]["group"] == calls[1]["group"]
    assert all("resume" not in call and call["settings"]["disable_git"] for call in calls)
    assert len(json.loads((tmp_path / "tracking.json").read_text())) == 2
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all("bad" not in row and "text" not in row for row in rows)
    assert all(run.exit_code == 0 for run in runs)


def test_disabled_tracking_still_writes_local_metrics(tmp_path):
    tracker = RunTracking(tmp_path, mode="disabled", project="rocky", entity=None,
                          identity={}, diagnostics={}, hourly_usd=1)
    tracker.log(3, {"eval/loss": 0.4})
    tracker.finish({"status": "failed"}, exit_code=1)
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["eval/loss"] == 0.4
