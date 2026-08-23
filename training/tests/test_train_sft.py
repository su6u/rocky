"""verify Gemma 4 rendering, masking, checkpoints, and SFT lineage"""

import json
from pathlib import Path

import pytest

from rocky_training.model_spec import load_model_spec
from rocky_training.paths import DEFAULT_GEMMA_E4B_IT, default_spec_path
from rocky_training.trainer_jsonl import load_trainer_jsonl
from rocky_training.train_sft import (
    SftTrainingResult,
    TrainSftError,
    build_conversation_dataset_rows,
    default_validation_dataset_path,
    gemma_messages_for_training,
    resolve_train_base_model,
    resolve_resume_checkpoint,
    run_train_sft,
    trainer_checkpoint_metric,
    validate_assistant_loss_mask,
    validate_chat_template,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeTokenizer:
    chat_template: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
    ) -> str | dict[str, list[int]]:
        assert add_generation_prompt is False
        rendered: list[str] = ["<bos>"]
        character_mask: list[int] = [0] * len(rendered[0])
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            turn = f"<|turn>{role}\n{message['content']}<turn|>\n"
            rendered.append(turn)
            character_mask.extend([1 if role == "model" else 0] * len(turn))
        text = "".join(rendered)
        if not tokenize:
            return text
        assert return_dict is True
        assert return_assistant_tokens_mask is True
        return {"input_ids": [ord(character) for character in text], "assistant_masks": character_mask}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token_id) for token_id in token_ids)


def test_default_validation_dataset_path_replaces_train_suffix() -> None:
    path = Path("exports/rocky-v1.train.jsonl")
    assert default_validation_dataset_path(path) == Path("exports/rocky-v1.holdout.jsonl")


def test_gemma_messages_preserve_native_system_turn() -> None:
    row = load_trainer_jsonl(FIXTURES / "smoke.train.jsonl", max_rows=1)[0]
    messages = gemma_messages_for_training(row)

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == "You are Rocky."
    assert messages[1]["content"].startswith("Pump seal leaks")
    assert '"spoken"' in messages[2]["content"]


def test_build_conversation_dataset_rows_preserves_ids() -> None:
    rows = load_trainer_jsonl(FIXTURES / "smoke.train.jsonl")
    dataset_rows = build_conversation_dataset_rows(rows)

    assert dataset_rows[0]["id"] == "smoke-1"
    assert dataset_rows[0]["messages"][-1]["role"] == "assistant"


class FakeBatchEncoding:
    """Mimic transformers BatchEncoding: has .get but is not a dict subclass."""

    def __init__(self, data: dict[str, list[int]]) -> None:
        self._data = data

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


class BatchEncodingTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
    ) -> str | FakeBatchEncoding:
        result = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
        )
        if isinstance(result, dict):
            return FakeBatchEncoding(result)
        return result


def test_validate_chat_template_uses_tokenizer_template() -> None:
    rows = load_trainer_jsonl(FIXTURES / "smoke.train.jsonl", max_rows=1)
    spec = load_model_spec(default_spec_path())
    rendered = validate_chat_template(FakeTokenizer(), rows, spec)

    assert "<|turn>system" in rendered
    assert "<|turn>user" in rendered
    assert "<|turn>model" in rendered
    assert "<start_of_turn>" not in rendered
    assert '"callbackId":null}<turn|>' in rendered
    assert "You are Rocky." in rendered


def test_validate_assistant_loss_mask_accepts_batch_encoding_mapping() -> None:
    rows = load_trainer_jsonl(FIXTURES / "smoke.train.jsonl", max_rows=1)
    messages = gemma_messages_for_training(rows[0])
    validate_assistant_loss_mask(BatchEncodingTokenizer(), messages)


def test_resolve_train_base_model_uses_checked_in_spec() -> None:
    spec = load_model_spec(default_spec_path())

    assert resolve_train_base_model(spec) == "google/gemma-4-31B-it"
    assert resolve_train_base_model(spec, "override/model") == "override/model"


def test_trainer_checkpoint_metric_maps_composite_gates_to_eval_loss() -> None:
    spec = load_model_spec(default_spec_path())

    assert spec.checkpoint_metric == "composite_gates"
    # Trainer loop can only optimize CE; persona/composite selection is post-train.
    assert trainer_checkpoint_metric(spec) == ("eval_loss", False)


def test_trainer_checkpoint_metric_keeps_eval_loss_when_configured() -> None:
    spec = load_model_spec(default_spec_path())
    eval_loss_spec = type(spec)(**{**spec.__dict__, "checkpoint_metric": "eval_loss"})

    assert trainer_checkpoint_metric(eval_loss_spec) == ("eval_loss", False)


def test_resolve_train_base_model_rejects_placeholder() -> None:
    spec = load_model_spec(default_spec_path())
    placeholder_spec = type(spec)(
        **{
            **spec.__dict__,
            "base_model": "PLACEHOLDER_VERIFY_UPSTREAM_GEMMA_E4B_IT",
        }
    )

    with pytest.raises(TrainSftError):
        resolve_train_base_model(placeholder_spec)


def test_resolve_resume_checkpoint_latest(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "checkpoint-10").mkdir(parents=True)
    (checkpoint_root / "checkpoint-50").mkdir(parents=True)

    resolved = resolve_resume_checkpoint(tmp_path, "latest")
    assert resolved is not None
    assert resolved.endswith("checkpoint-50")


def test_resolve_resume_checkpoint_missing_latest(tmp_path: Path) -> None:
    with pytest.raises(TrainSftError, match="no checkpoints found"):
        resolve_resume_checkpoint(tmp_path, "latest")


def test_run_train_sft_dry_run_writes_manifest(tmp_path: Path) -> None:
    train_path = tmp_path / "rocky-v1.train.jsonl"
    holdout_path = tmp_path / "rocky-v1.holdout.jsonl"
    fixture_text = (FIXTURES / "smoke.train.jsonl").read_text(encoding="utf-8")
    train_path.write_text(fixture_text, encoding="utf-8")
    holdout_path.write_text(fixture_text, encoding="utf-8")

    manifest = run_train_sft(
        spec_path=default_spec_path(),
        dataset_path=train_path,
        output_dir=tmp_path / "run",
        base_model=DEFAULT_GEMMA_E4B_IT,
        max_rows=1,
        max_validation_rows=1,
        dry_run=True,
        tokenizer_loader=lambda _model: FakeTokenizer(),
    )

    assert manifest["kind"] == "train-sft"
    assert manifest["assistantOnlyLoss"] is True
    assert manifest["optimizer"]["saveSteps"] == 50
    assert manifest["optimizer"]["checkpointMetric"] == "composite_gates"
    assert manifest["optimizer"]["warmupRatio"] == 0.05
    assert manifest["adapter"]["dropout"] == 0.05
    assert manifest["checkpointDir"].endswith("checkpoints")
    assert manifest["trainRowCount"] == 1
    assert manifest["validationRowCount"] == 1
    saved = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert "<|turn>user" in saved["renderedChatTemplateSample"]


def test_run_train_sft_fake_runner_records_validation_rows(tmp_path: Path) -> None:
    train_path = tmp_path / "rocky-v1.train.jsonl"
    holdout_path = tmp_path / "rocky-v1.holdout.jsonl"
    fixture_text = (FIXTURES / "smoke.train.jsonl").read_text(encoding="utf-8")
    train_path.write_text(fixture_text, encoding="utf-8")
    holdout_path.write_text(fixture_text, encoding="utf-8")

    def fake_runner(**kwargs: object) -> SftTrainingResult:
        assert len(kwargs["train_rows"]) == 2
        assert len(kwargs["validation_rows"]) == 2
        return SftTrainingResult(
            base_model=DEFAULT_GEMMA_E4B_IT,
            adapter_dir=str(tmp_path / "run" / "adapter"),
            train_loss=0.9,
            eval_loss=1.1,
            best_metric=1.0,
            global_step=12,
        )

    manifest = run_train_sft(
        spec_path=default_spec_path(),
        dataset_path=train_path,
        output_dir=tmp_path / "run",
        base_model=DEFAULT_GEMMA_E4B_IT,
        train_runner=fake_runner,
    )

    assert manifest["dryRun"] is False
    assert manifest["evalLoss"] == 1.1
    assert manifest["globalStep"] == 12
