"""train Rocky adapters with assistant-only loss and release lineage"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any, Callable

from rocky_training.model_spec import ModelSpec, load_model_spec
from rocky_training.paths import training_root
from rocky_training.release_guard import DataReleaseError, require_approved_data_release
from rocky_training.response_parse import parse_model_output
from rocky_training.trainer_jsonl import TrainerExportRow, load_trainer_jsonl, write_json


class TrainSftError(Exception):
    pass


@dataclass(frozen=True)
class SftTrainingResult:
    base_model: str
    adapter_dir: str
    train_loss: float | None
    eval_loss: float | None
    best_metric: float | None
    global_step: int


def default_validation_dataset_path(dataset_path: Path) -> Path:
    if dataset_path.name.endswith(".train.jsonl"):
        return dataset_path.with_name(dataset_path.name.replace(".train.jsonl", ".holdout.jsonl"))
    return dataset_path.with_suffix(".holdout.jsonl")


def resolve_train_base_model(spec: ModelSpec, override: str | None = None) -> str:
    base_model = override or spec.base_model
    if base_model.startswith("PLACEHOLDER_"):
        raise TrainSftError(
            "base_model is still a placeholder; pass --base-model with the verified Gemma train model id"
        )
    return base_model


def resolve_resume_checkpoint(output_dir: Path, resume: str | None) -> str | None:
    if resume is None:
        return None

    if resume == "latest":
        checkpoint_root = output_dir / "checkpoints"
        checkpoints = sorted(
            checkpoint_root.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if not checkpoints:
            raise TrainSftError(f"no checkpoints found in {checkpoint_root}")
        return str(checkpoints[-1])

    path = Path(resume)
    if not path.is_dir():
        raise TrainSftError(f"resume checkpoint not found: {path}")
    return str(path)


def gemma_messages_for_training(row: TrainerExportRow) -> list[dict[str, str]]:
    messages = [{"role": message.role, "content": message.content} for message in row.messages]

    if not messages:
        raise TrainSftError(f"{row.id}: row has no messages")
    if not any(message["role"] == "user" for message in messages):
        raise TrainSftError(f"{row.id}: row has no user message")
    if messages[-1]["role"] != "assistant":
        raise TrainSftError(f"{row.id}: final training message must be assistant")

    return messages


def build_conversation_dataset_rows(rows: list[TrainerExportRow]) -> list[dict[str, Any]]:
    return [{"id": row.id, "messages": gemma_messages_for_training(row)} for row in rows]


def is_gemma4_template(spec: ModelSpec | None) -> bool:
    if spec is None:
        return False
    template = spec.chat_template.lower()
    return template == "gemma4" or "gemma4" in template


def default_gemma4_training_template_path() -> Path:
    return training_root() / "templates" / "gemma4_training.jinja"


def resolve_chat_template_path(spec: ModelSpec) -> Path | None:
    if spec.chat_template == "gemma":
        return None
    if spec.chat_template == "gemma4":
        return default_gemma4_training_template_path()

    path = Path(spec.chat_template)
    if not path.is_absolute():
        path = training_root().parent / path
    return path


def load_training_chat_template(spec: ModelSpec) -> str | None:
    path = resolve_chat_template_path(spec)
    if path is None:
        return None
    if not path.is_file():
        raise TrainSftError(f"chat template not found: {path}")
    return path.read_text(encoding="utf-8")


def apply_training_chat_template(tokenizer: Any, spec: ModelSpec) -> None:
    template = load_training_chat_template(spec)
    if template is not None:
        tokenizer.chat_template = template


def validate_gemma4_rendered_training_sample(rendered: str) -> None:
    required = ("<|turn>system", "<|turn>user", "<|turn>model", "<turn|>")
    missing = [token for token in required if token not in rendered]
    if missing:
        raise TrainSftError(f"Gemma 4 training template missing required tokens: {', '.join(missing)}")
    if "<start_of_turn>" in rendered or "<end_of_turn>" in rendered:
        raise TrainSftError("Gemma 4 training template rendered deprecated Gemma 1/2 turn tokens")
    assistant_start = rendered.find("<|turn>model")
    turn_close = rendered.find("<turn|>", assistant_start)
    response_start = rendered.find("{", assistant_start, turn_close)
    if assistant_start == -1 or response_start == -1 or turn_close == -1:
        raise TrainSftError("Gemma 4 assistant span must include a response object before <turn|>")
    response = rendered[response_start:turn_close].strip()
    if parse_model_output(response).response_json is None:
        raise TrainSftError("Gemma 4 assistant span must contain a valid Rocky v1 response")


def _as_token_id_list(value: Any) -> list[int] | None:
    """Normalize HF BatchEncoding / tensor / nested-list tokenize outputs to list[int]."""
    if value is None:
        return None
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, str)):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, (int, bool)) for item in value):
        return None
    return [int(item) for item in value]


def validate_assistant_loss_mask(tokenizer: Any, messages: list[dict[str, str]]) -> None:
    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception as error:  # pragma: no cover - depends on model tokenizer
        raise TrainSftError("Gemma 4 template failed to produce an assistant loss mask") from error

    # BatchEncoding is Mapping/UserDict, not always isinstance(..., dict)
    if not hasattr(tokenized, "get"):
        raise TrainSftError("Gemma 4 template did not return input_ids and assistant_masks")

    input_ids = _as_token_id_list(tokenized.get("input_ids"))
    assistant_mask = _as_token_id_list(tokenized.get("assistant_masks"))
    if input_ids is None or assistant_mask is None:
        raise TrainSftError("Gemma 4 template did not return input_ids and assistant_masks")
    if len(input_ids) != len(assistant_mask) or not any(assistant_mask):
        raise TrainSftError("Gemma 4 assistant loss mask is empty or misaligned")
    if all(assistant_mask):
        raise TrainSftError("Gemma 4 assistant loss mask incorrectly includes the whole conversation")

    masked_ids = [token_id for token_id, included in zip(input_ids, assistant_mask) if included]
    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
    expected_assistant_turns = sum(message["role"] == "assistant" for message in messages)
    if masked_text.count('"spoken"') != expected_assistant_turns:
        raise TrainSftError("Gemma 4 assistant loss mask does not cover every response object")


def validate_chat_template(
    tokenizer: Any,
    rows: list[TrainerExportRow],
    spec: ModelSpec | None = None,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TrainSftError("tokenizer does not support apply_chat_template")
    if not rows:
        raise TrainSftError("dataset contains no rows")

    messages = gemma_messages_for_training(rows[0])
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as error:  # pragma: no cover - depends on model tokenizer
        raise TrainSftError("Gemma chat template failed for exported training messages") from error

    if not isinstance(rendered, str) or len(rendered) == 0:
        raise TrainSftError("Gemma chat template rendered an empty prompt")
    if is_gemma4_template(spec):
        validate_gemma4_rendered_training_sample(rendered)
        validate_assistant_loss_mask(tokenizer, messages)
    return rendered


def _require_train_dependencies() -> None:
    missing: list[str] = []
    for module in ("datasets", "peft", "torch", "transformers", "trl"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise TrainSftError(
            "missing training dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install -e 'training/.[train]'"
        )


def _metric_from_history(history: list[dict[str, Any]], key: str) -> float | None:
    values = [entry[key] for entry in history if isinstance(entry.get(key), (int, float))]
    return float(values[-1]) if values else None


def trainer_checkpoint_metric(spec: ModelSpec) -> tuple[str, bool]:
    # composite_gates is selected post-train via select-checkpoint; Trainer loop uses eval_loss.
    if spec.checkpoint_metric == "composite_gates":
        return ("eval_loss", False)
    return (spec.checkpoint_metric, False)


def current_wandb_run() -> dict[str, str] | None:
    try:
        import wandb  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    run = getattr(wandb, "run", None)
    if run is None:
        return None
    run_id = getattr(run, "id", None)
    url = getattr(run, "url", None)
    project = getattr(run, "project", None)
    result: dict[str, str] = {}
    if isinstance(run_id, str) and run_id:
        result["id"] = run_id
    if isinstance(url, str) and url:
        result["url"] = url
    if isinstance(project, str) and project:
        result["project"] = project
    return result or None


def run_sft_training(
    *,
    spec: ModelSpec,
    base_model: str,
    train_rows: list[TrainerExportRow],
    validation_rows: list[TrainerExportRow],
    output_dir: Path,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int | None = None,
    report_to: list[str] | None = None,
    resume_from_checkpoint: str | None = None,
) -> SftTrainingResult:
    _require_train_dependencies()

    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    from rocky_training.model_load import ModelLoadError, load_rocky_train_model

    try:
        loaded = load_rocky_train_model(spec=spec, base_model=base_model, mode="train")
    except ModelLoadError as error:
        raise TrainSftError(str(error)) from error

    tokenizer = loaded.tokenizer
    apply_training_chat_template(tokenizer, spec)
    validate_chat_template(tokenizer, train_rows, spec)

    train_dataset = Dataset.from_list(build_conversation_dataset_rows(train_rows))
    eval_dataset = Dataset.from_list(build_conversation_dataset_rows(validation_rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"

    resolved_gradient_accumulation_steps = gradient_accumulation_steps or max(
        1,
        math.ceil(spec.optimizer.effective_batch_size / per_device_train_batch_size),
    )

    checkpoint_metric, greater_is_better = trainer_checkpoint_metric(spec)
    warmup_args: dict[str, Any] = {}
    if spec.optimizer.warmup_ratio is not None:
        warmup_args["warmup_ratio"] = spec.optimizer.warmup_ratio
    else:
        warmup_args["warmup_steps"] = spec.optimizer.warmup_steps
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=spec.optimizer.max_epochs,
        max_length=spec.sequence.max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=resolved_gradient_accumulation_steps,
        learning_rate=spec.optimizer.learning_rate,
        lr_scheduler_type=spec.optimizer.scheduler,
        weight_decay=spec.optimizer.weight_decay,
        bf16=spec.train_precision == "bf16",
        fp16=spec.train_precision == "fp16",
        gradient_checkpointing=True,
        logging_steps=spec.optimizer.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch" if spec.optimizer.early_stopping else "steps",
        save_steps=spec.optimizer.save_steps,
        save_total_limit=spec.optimizer.save_total_limit,
        load_best_model_at_end=spec.optimizer.early_stopping,
        metric_for_best_model=checkpoint_metric,
        greater_is_better=greater_is_better,
        assistant_only_loss=True,
        packing=False,
        report_to=report_to or [],
        remove_unused_columns=False,
        **warmup_args,
    )

    callbacks = []
    if spec.optimizer.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=1))
    trainer = SFTTrainer(
        model=loaded.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=loaded.lora_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    history = trainer.state.log_history
    train_loss = _metric_from_history(history, "loss")
    eval_loss = _metric_from_history(history, "eval_loss")
    best_metric = (
        float(trainer.state.best_metric)
        if isinstance(trainer.state.best_metric, (int, float))
        else None
    )
    if train_loss is None and isinstance(train_output.metrics.get("train_loss"), (int, float)):
        train_loss = float(train_output.metrics["train_loss"])

    return SftTrainingResult(
        base_model=base_model,
        adapter_dir=str(adapter_dir),
        train_loss=train_loss,
        eval_loss=eval_loss,
        best_metric=best_metric,
        global_step=int(trainer.state.global_step),
    )


def build_train_sft_manifest(
    *,
    spec: ModelSpec,
    dataset_path: Path,
    validation_dataset_path: Path,
    output_dir: Path,
    base_model: str,
    train_rows: list[TrainerExportRow],
    validation_rows: list[TrainerExportRow],
    rendered_chat_template_sample: str,
    training: SftTrainingResult | None,
    dry_run: bool,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "train-sft",
        "dryRun": dry_run,
        "specId": spec.id,
        "baseModel": base_model,
        "chatTemplate": spec.chat_template,
        "assistantOnlyLoss": True,
        "adapter": {
            "method": spec.adapter.method,
            "trainQuantization": spec.quantization.train,
            "rank": spec.adapter.rank,
            "alpha": spec.adapter.alpha,
            "dropout": spec.adapter.dropout,
            "targetModules": list(spec.adapter.target_modules),
        },
        "optimizer": {
            "learningRate": spec.optimizer.learning_rate,
            "scheduler": spec.optimizer.scheduler,
            "warmupSteps": spec.optimizer.warmup_steps,
            "warmupRatio": spec.optimizer.warmup_ratio,
            "weightDecay": spec.optimizer.weight_decay,
            "effectiveBatchSize": spec.optimizer.effective_batch_size,
            "maxEpochs": spec.optimizer.max_epochs,
            "earlyStopping": spec.optimizer.early_stopping,
            "checkpointMetric": spec.checkpoint_metric,
            "saveSteps": spec.optimizer.save_steps,
            "saveTotalLimit": spec.optimizer.save_total_limit,
            "loggingSteps": spec.optimizer.logging_steps,
        },
        "checkpointDir": str(output_dir / "checkpoints"),
        "resumeFromCheckpoint": resume_from_checkpoint,
        "wandb": current_wandb_run(),
        "wandbProject": os.environ.get("WANDB_PROJECT"),
        "datasetPath": str(dataset_path),
        "validationDatasetPath": str(validation_dataset_path),
        "trainRowCount": len(train_rows),
        "validationRowCount": len(validation_rows),
        "outputDir": str(output_dir),
        "renderedChatTemplateSample": rendered_chat_template_sample,
        "adapterDir": training.adapter_dir if training else str(output_dir / "adapter"),
        "trainLoss": training.train_loss if training else None,
        "evalLoss": training.eval_loss if training else None,
        "bestMetric": training.best_metric if training else None,
        "globalStep": training.global_step if training else 0,
        "finishedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


def run_train_sft(
    *,
    spec_path: Path,
    dataset_path: Path,
    output_dir: Path,
    validation_dataset_path: Path | None = None,
    base_model: str | None = None,
    max_rows: int = 0,
    max_validation_rows: int = 0,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int | None = None,
    report_to: list[str] | None = None,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    train_runner: Callable[..., SftTrainingResult] | None = None,
    tokenizer_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    spec = load_model_spec(spec_path)
    resolved_base_model = resolve_train_base_model(spec, base_model)
    resolved_validation_path = validation_dataset_path or default_validation_dataset_path(dataset_path)
    resolved_resume = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)

    if not dry_run and train_runner is None:
        try:
            require_approved_data_release(dataset_path, dataset_kind="sft")
        except DataReleaseError as error:
            raise TrainSftError(str(error)) from error

    if not resolved_validation_path.is_file():
        raise TrainSftError(f"validation dataset not found: {resolved_validation_path}")

    train_rows = load_trainer_jsonl(dataset_path, max_rows=max_rows)
    validation_rows = load_trainer_jsonl(resolved_validation_path, max_rows=max_validation_rows)

    if tokenizer_loader:
        tokenizer = tokenizer_loader(resolved_base_model)
        apply_training_chat_template(tokenizer, spec)
        rendered_sample = validate_chat_template(tokenizer, train_rows, spec)
    else:
        rendered_sample = "dry-run skipped tokenizer load" if dry_run else ""

    training: SftTrainingResult | None = None
    if not dry_run:
        runner = train_runner or run_sft_training
        training = runner(
            spec=spec,
            base_model=resolved_base_model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            report_to=report_to,
            resume_from_checkpoint=resolved_resume,
        )
        if not rendered_sample and tokenizer_loader is None:
            rendered_sample = "see trainer tokenizer chat template"

    manifest = build_train_sft_manifest(
        spec=spec,
        dataset_path=dataset_path,
        validation_dataset_path=resolved_validation_path,
        output_dir=output_dir,
        base_model=resolved_base_model,
        train_rows=train_rows,
        validation_rows=validation_rows,
        rendered_chat_template_sample=rendered_sample,
        training=training,
        dry_run=dry_run,
        resume_from_checkpoint=resolved_resume,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest
