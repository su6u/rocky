import json

import pytest

from rocky.cli import build_parser
from rocky.config import CONFIG, load_config, tokenizer_location
from rocky.training import prepare_run, run_training


def test_cli_revision_does_not_use_shared_temporary_file():
    parsed = build_parser().parse_args(["train-sft", "--resume", "run/checkpoint-10", "--execute", "--hourly-usd", "1.59"])
    assert parsed.resume.name == "checkpoint-10"
    assert parsed.curriculum == "core"
    assert load_config(revision="a" * 40)["revision"] == "a" * 40
    assert json.loads(CONFIG.read_text())["revision"] != "a" * 40


def test_distributed_launch_rejected_before_loading_model(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-process"):
        run_training()


def test_resume_rejects_incomplete_or_changed_checkpoint(tmp_path):
    run = tmp_path / "run"
    identity = {"model": "pinned"}
    prepare_run(run, identity, None)
    checkpoint = run / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":1}')
    with pytest.raises(ValueError, match="incomplete"):
        prepare_run(run, identity, checkpoint)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        (checkpoint / name).write_text("test-only")
    with pytest.raises(ValueError, match="resume rejected"):
        prepare_run(run, {"model": "changed"}, checkpoint)
    assert prepare_run(run, identity, checkpoint)["identity"] == identity


def test_no_quantized_or_smaller_config(tmp_path):
    config = json.loads(CONFIG.read_text())
    path = tmp_path / "config.json"
    for key, value in (("quantization", "nf4"), ("model_id", "google/gemma-4-31B-it"), ("model_id", "google/gemma-4-26B-A4B-it"), ("max_hourly_usd", 2.01)):
        altered = {**config, key: value}
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            load_config(path)


def test_audited_local_tokenizer_matches_pinned_model():
    source, kwargs = tokenizer_location(load_config())
    assert source.endswith(".cache/gemma-e4b-tokenizer")
    assert kwargs == {"local_files_only": True}
