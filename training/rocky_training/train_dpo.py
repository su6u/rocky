"""train preferences only from an approved, reviewed release"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rocky_training.eval_gates import has_single_response_object, response_is_valid
from rocky_training.model_spec import ModelSpec, load_model_spec
from rocky_training.paths import default_system_prompt_path
from rocky_training.release_guard import DataReleaseError, require_approved_data_release
from rocky_training.train_sft import (
    TrainSftError,
    apply_training_chat_template,
    is_gemma4_template,
    resolve_train_base_model,
)
from rocky_training.trainer_jsonl import write_json


class TrainDpoError(Exception):
    pass


@dataclass(frozen=True)
class PreferenceRow:
    id: str
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class DpoTrainingResult:
    adapter_dir: str
    train_loss: float | None
    global_step: int


def load_system_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path is not None else default_system_prompt_path()
    if not prompt_path.is_file():
        raise TrainDpoError(f"system prompt file not found: {prompt_path}")
    # Match @rocky/prompt SYSTEM_PROMPT (sync-contracts adds a trailing newline).
    return prompt_path.read_text(encoding="utf-8").rstrip("\n")


def load_preference_jsonl(path: str | Path, *, max_rows: int = 0) -> list[PreferenceRow]:
    rows: list[PreferenceRow] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    file_path = Path(path)
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainDpoError(f"line {line_number}: invalid json") from error
        if not isinstance(parsed, dict):
            raise TrainDpoError(f"line {line_number}: row must be an object")
        row_id = parsed.get("id")
        prompt = parsed.get("prompt")
        chosen = parsed.get("chosen")
        rejected = parsed.get("rejected")
        if not isinstance(row_id, str) or not row_id:
            raise TrainDpoError(f"line {line_number}: id must be a non-empty string")
        for field_name, value in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)):
            if not isinstance(value, str) or not value:
                raise TrainDpoError(f"line {line_number}: {field_name} must be a non-empty string")
        if chosen == rejected:
            raise TrainDpoError(f"line {line_number}: chosen and rejected must differ")
        for field_name, completion in (("chosen", chosen), ("rejected", rejected)):
            if not has_single_response_object(completion) or not response_is_valid(completion):
                raise TrainDpoError(
                    f"line {line_number}: {field_name} must be one valid Rocky v1 response object"
                )
        normalized_prompt = " ".join(prompt.lower().split())
        if row_id in seen_ids:
            raise TrainDpoError(f"line {line_number}: duplicate preference id")
        if normalized_prompt in seen_prompts:
            raise TrainDpoError(f"line {line_number}: duplicate preference prompt")
        seen_ids.add(row_id)
        seen_prompts.add(normalized_prompt)
        rows.append(PreferenceRow(id=row_id, prompt=prompt, chosen=chosen, rejected=rejected))
        if max_rows > 0 and len(rows) >= max_rows:
            break
    if not rows:
        raise TrainDpoError("preference dataset contains no rows")
    return rows


def conversational_preference_row(row: PreferenceRow, system_prompt: str) -> dict[str, Any]:
    return {
        "id": row.id,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row.prompt},
        ],
        "chosen": [{"role": "assistant", "content": row.chosen}],
        "rejected": [{"role": "assistant", "content": row.rejected}],
    }


def build_preference_dataset_rows(
    rows: list[PreferenceRow],
    *,
    spec: ModelSpec | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    if spec is not None and is_gemma4_template(spec):
        resolved_system_prompt = system_prompt if system_prompt is not None else load_system_prompt()
        return [conversational_preference_row(row, resolved_system_prompt) for row in rows]
    return [
        {"id": row.id, "prompt": row.prompt, "chosen": row.chosen, "rejected": row.rejected}
        for row in rows
    ]


def _require_dpo_dependencies() -> None:
    missing: list[str] = []
    for module in ("datasets", "peft", "torch", "transformers", "trl"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise TrainDpoError(
            "missing DPO dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install -e 'training/.[train]'"
        )


def _metric_from_history(history: list[dict[str, Any]], key: str) -> float | None:
    values = [entry[key] for entry in history if isinstance(entry.get(key), (int, float))]
    return float(values[-1]) if values else None


def run_dpo_training(
    *,
    spec: ModelSpec,
    base_model: str,
    rows: list[PreferenceRow],
    output_dir: Path,
    beta: float,
    learning_rate: float,
    sft_adapter_dir: Path,
    report_to: list[str] | None = None,
    system_prompt_path: Path | None = None,
) -> DpoTrainingResult:
    _require_dpo_dependencies()

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    from rocky_training.model_load import ModelLoadError, load_rocky_train_model

    if not sft_adapter_dir.is_dir():
        raise TrainDpoError(f"SFT adapter dir not found: {sft_adapter_dir}")
    try:
        loaded = load_rocky_train_model(
            spec=spec,
            base_model=base_model,
            mode="train",
            sft_adapter_dir=str(sft_adapter_dir),
        )
    except ModelLoadError as error:
        raise TrainDpoError(str(error)) from error

    tokenizer = loaded.tokenizer
    apply_training_chat_template(tokenizer, spec)
    system_prompt = load_system_prompt(system_prompt_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    training_args = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        beta=beta,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=max(1, spec.optimizer.effective_batch_size),
        max_length=spec.sequence.max_length,
        num_train_epochs=1,
        bf16=spec.train_precision == "bf16",
        fp16=spec.train_precision == "fp16",
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=report_to or [],
    )
    trainer = DPOTrainer(
        model=loaded.model,
        args=training_args,
        train_dataset=Dataset.from_list(
            build_preference_dataset_rows(rows, spec=spec, system_prompt=system_prompt)
        ),
        processing_class=tokenizer,
        peft_config=loaded.lora_config,
    )
    output = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    train_loss = _metric_from_history(trainer.state.log_history, "loss")
    if train_loss is None and isinstance(output.metrics.get("train_loss"), (int, float)):
        train_loss = float(output.metrics["train_loss"])
    return DpoTrainingResult(
        adapter_dir=str(adapter_dir),
        train_loss=train_loss,
        global_step=int(trainer.state.global_step),
    )


def run_train_dpo(
    *,
    spec_path: Path,
    dataset_path: Path,
    output_dir: Path,
    base_model: str | None = None,
    max_rows: int = 0,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    sft_adapter_dir: Path | None = None,
    report_to: list[str] | None = None,
    system_prompt_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    spec = load_model_spec(spec_path)
    try:
        resolved_base_model = resolve_train_base_model(spec, base_model)
    except TrainSftError as error:
        raise TrainDpoError(str(error)) from error
    if not dry_run:
        try:
            require_approved_data_release(dataset_path, dataset_kind="preference")
        except DataReleaseError as error:
            raise TrainDpoError(str(error)) from error
    rows = load_preference_jsonl(dataset_path, max_rows=max_rows)
    resolved_sft_adapter_dir = sft_adapter_dir or Path(spec.artifacts.adapter_dir)
    if not dry_run and not resolved_sft_adapter_dir.is_dir():
        raise TrainDpoError(f"SFT adapter dir not found: {resolved_sft_adapter_dir}")

    training: DpoTrainingResult | None = None
    if not dry_run:
        training = run_dpo_training(
            spec=spec,
            base_model=resolved_base_model,
            rows=rows,
            output_dir=output_dir,
            beta=beta,
            learning_rate=learning_rate,
            sft_adapter_dir=resolved_sft_adapter_dir,
            report_to=report_to,
            system_prompt_path=system_prompt_path,
        )

    manifest = {
        "kind": "train-dpo",
        "dryRun": dry_run,
        "specId": spec.id,
        "baseModel": resolved_base_model,
        "datasetPath": str(dataset_path),
        "sftAdapterDir": str(resolved_sft_adapter_dir),
        "chatTemplate": spec.chat_template,
        "preferenceRowCount": len(rows),
        "beta": beta,
        "learningRate": learning_rate,
        "adapterDir": training.adapter_dir if training else str(output_dir / "adapter"),
        "trainLoss": training.train_loss if training else None,
        "globalStep": training.global_step if training else 0,
        "finishedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
