from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from rocky_training.model_spec import ModelSpec, load_model_spec
from rocky_training.trainer_jsonl import (
    TrainerExportRow,
    format_smoke_example,
    load_trainer_jsonl,
    sha256_text,
    write_json,
)

SMOKE_FALLBACK_MODEL = "sshleifer/tiny-gpt2"
FIXED_SMOKE_PROMPT = "Pump seal leaks. What should we do?"


@dataclass(frozen=True)
class SmokeTrainingResult:
    loss_start: float
    loss_end: float
    before_output: str
    after_output: str
    base_model: str


def resolve_smoke_base_model(_spec: ModelSpec, override: str | None = None) -> str:
    if override:
        return override
    return SMOKE_FALLBACK_MODEL


def build_smoke_manifest(
    *,
    spec: ModelSpec,
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    prompt_sample: str,
    label_sample: str,
    training: SmokeTrainingResult,
    max_rows: int,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "kind": "smoke-sft",
        "specId": spec.id,
        "datasetPath": str(dataset_path),
        "outputDir": str(output_dir),
        "baseModel": base_model,
        "maxRows": max_rows,
        "maxSteps": max_steps,
        "promptSample": prompt_sample,
        "labelSample": label_sample,
        "promptHash": sha256_text(prompt_sample.split("\n")[0]),
        "lossStart": training.loss_start,
        "lossEnd": training.loss_end,
        "fixedPrompt": FIXED_SMOKE_PROMPT,
        "beforeOutput": training.before_output,
        "afterOutput": training.after_output,
        "finishedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


def _require_train_dependencies() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from peft import LoraConfig  # noqa: F401
        from trl import SFTTrainer  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "smoke-sft requires train extras: pip install -e '.[train]'"
        ) from error


def _format_rows_for_training(rows: list[TrainerExportRow]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for row in rows:
        prompt, label = format_smoke_example(row)
        formatted.append({"text": f"{prompt} {label}"})
    return formatted


def _generate_output(model: Any, tokenizer: Any, prompt: str) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if generated.startswith(prompt):
        return generated[len(prompt) :].strip()
    return generated.strip()


def run_smoke_training(
    *,
    base_model: str,
    rows: list[TrainerExportRow],
    output_dir: Path,
    max_steps: int,
) -> SmokeTrainingResult:
    _require_train_dependencies()

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    prompt_sample, label_sample = format_smoke_example(rows[0])
    print("smoke prompt sample:")
    print(prompt_sample)
    print("smoke label sample:")
    print(label_sample)

    eval_prompt = f"You are Rocky.\nUser: {FIXED_SMOKE_PROMPT}\nAssistant:"
    before_output = _generate_output(model, tokenizer, eval_prompt)

    dataset = Dataset.from_list(_format_rows_for_training(rows))
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        learning_rate=5e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    loss_start = float(metrics.get("train_loss", metrics.get("loss", 0.0)))
    loss_end = float(metrics.get("train_loss", metrics.get("loss", 0.0)))
    history = trainer.state.log_history
    if history:
        losses = [entry["loss"] for entry in history if "loss" in entry]
        if losses:
            loss_start = float(losses[0])
            loss_end = float(losses[-1])

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    after_output = _generate_output(model, tokenizer, eval_prompt)

    return SmokeTrainingResult(
        loss_start=loss_start,
        loss_end=loss_end,
        before_output=before_output,
        after_output=after_output,
        base_model=base_model,
    )


def run_smoke_sft(
    *,
    spec_path: Path,
    dataset_path: Path,
    output_dir: Path,
    max_rows: int = 4,
    max_steps: int = 20,
    base_model: str | None = None,
    train_runner: Callable[..., SmokeTrainingResult] | None = None,
) -> dict[str, Any]:
    spec = load_model_spec(spec_path)
    rows = load_trainer_jsonl(dataset_path, max_rows=max_rows)
    prompt_sample, label_sample = format_smoke_example(rows[0])
    resolved_base_model = resolve_smoke_base_model(spec, base_model)

    runner = train_runner or run_smoke_training
    training = runner(
        base_model=resolved_base_model,
        rows=rows,
        output_dir=output_dir,
        max_steps=max_steps,
    )

    manifest = build_smoke_manifest(
        spec=spec,
        dataset_path=dataset_path,
        output_dir=output_dir,
        base_model=resolved_base_model,
        prompt_sample=prompt_sample,
        label_sample=label_sample,
        training=training,
        max_rows=max_rows,
        max_steps=max_steps,
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "prompts.json",
        {
            "fixedPrompt": FIXED_SMOKE_PROMPT,
            "beforeOutput": training.before_output,
            "afterOutput": training.after_output,
        },
    )
    return manifest
