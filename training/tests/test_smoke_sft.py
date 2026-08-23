"""verify the minimal non-GPU SFT smoke interface"""

import json
from pathlib import Path

from rocky_training.model_spec import load_model_spec
from rocky_training.paths import default_spec_path
from rocky_training.smoke_sft import (
    SMOKE_FALLBACK_MODEL,
    SmokeTrainingResult,
    resolve_smoke_base_model,
    run_smoke_sft,
)
from rocky_training.trainer_jsonl import format_smoke_example, load_trainer_jsonl


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_trainer_jsonl_reads_fixture() -> None:
    rows = load_trainer_jsonl(FIXTURES / "smoke.train.jsonl")
    assert len(rows) == 2
    assert rows[0].id == "smoke-1"


def test_format_smoke_example_splits_prompt_and_label() -> None:
    rows = load_trainer_jsonl(FIXTURES / "smoke.train.jsonl", max_rows=1)
    prompt, label = format_smoke_example(rows[0])
    assert "Pump seal leaks" in prompt
    assert '"spoken"' in label


def test_resolve_smoke_base_model_uses_tiny_fallback_by_default() -> None:
    spec = load_model_spec(default_spec_path())
    assert resolve_smoke_base_model(spec) == SMOKE_FALLBACK_MODEL


def test_load_trainer_jsonl_max_rows_zero_loads_all() -> None:
    rows = load_trainer_jsonl(FIXTURES / "smoke.train.jsonl", max_rows=0)
    assert len(rows) == 2


def test_run_smoke_sft_writes_manifest(tmp_path: Path) -> None:
    def fake_train(**_kwargs: object) -> SmokeTrainingResult:
        return SmokeTrainingResult(
            loss_start=1.2,
            loss_end=0.3,
            before_output="before",
            after_output="after",
            base_model=SMOKE_FALLBACK_MODEL,
        )

    manifest = run_smoke_sft(
        spec_path=default_spec_path(),
        dataset_path=FIXTURES / "smoke.train.jsonl",
        output_dir=tmp_path,
        max_rows=2,
        max_steps=5,
        train_runner=fake_train,
    )

    assert manifest["kind"] == "smoke-sft"
    assert manifest["lossEnd"] < manifest["lossStart"]
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "prompts.json").is_file()
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["beforeOutput"] == "before"
    assert saved["afterOutput"] == "after"
