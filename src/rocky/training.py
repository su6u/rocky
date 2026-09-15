"""Single-GPU, unquantized Gemma 4 adaptation with fixed SFT references."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

from rocky.analysis import dataset_analysis, token_diagnostics
from rocky.config import CONFIG, accumulation, load_config, tokenizer_location
from rocky.data import DATA, digest, read_jsonl, verify_export
from rocky.tokenization import ReplyCollator, encode_conversations, encode_reply
from rocky.tracking import RunTracking, check_tracking, tracking_callback

OBJECTIVES = ("sft", "dpo", "dpo-norm", "simpo")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def weight_manifest(directory: Path) -> dict:
    files = sorted(directory.glob("*.safetensors"))
    if not files or not (directory / "adapter_config.json").is_file():
        raise ValueError("run output must contain safetensors adapter weights and adapter config")
    result = {}
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        result[path.name] = hasher.hexdigest()
    return result


def sft_origin(directory: Path, config: dict) -> dict:
    record = json.loads((directory / "run.json").read_text())
    if record.get("status") != "completed" or record.get("identity", {}).get("objective") != "sft":
        raise ValueError("preference tuning requires a completed SFT run")
    old = record["identity"]["config"]
    for key in ("model_id", "revision", "precision", "quantization", "update", "lora_rank", "lora_alpha"):
        if old[key] != config[key]:
            raise ValueError("SFT model identity differs from preference configuration")
    # Preference authoring may change after SFT. Compare only actual SFT inputs,
    # not an unrelated preference file or evidence-document timestamp.
    for name, expected in record["identity"]["data_inputs"].items():
        if digest(DATA / "exports" / name) != expected:
            raise ValueError("SFT run used different supervised data")
    if record.get("weights") != weight_manifest(directory / "model"):
        raise ValueError("SFT weights changed after the run completed")
    return record


def prepare_run(output: Path, identity: dict, resume: Path | None) -> dict:
    if resume is not None:
        if not resume.is_dir() or resume.resolve().parent != output.resolve():
            raise ValueError("resume checkpoint must be a direct child of this run directory")
        state = json.loads((resume / "trainer_state.json").read_text())
        if not isinstance(state.get("global_step"), int) or state["global_step"] < 1:
            raise ValueError("resume checkpoint has no completed training step")
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
            if not (resume / name).is_file():
                raise ValueError(f"resume checkpoint is incomplete: {name}")
        record = json.loads((output / "run.json").read_text())
        if record["identity"] != identity:
            raise ValueError("resume rejected: model, data, code, tokenizer, or configuration changed")
        return record
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty; use verified resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    record = {"identity": identity, "status": "prepared", "persona_accepted": False}
    write_json(output / "run.json", record)
    return record


def load_policy(config, sft_run=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForMultimodalLM

    model, loading = AutoModelForMultimodalLM.from_pretrained(
        config["model_id"], revision=config["revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
    )
    # Upstream suppresses unused projections in E4B's KV-sharing layers.
    # Tied output embeddings need no separate checkpoint tensor. Any remaining
    # mismatch must fail rather than train randomly initialized missing weights.
    missing = set(loading.get("missing_keys", [])) - {"lm_head.weight"}
    if missing or any(loading.get(key) for key in ("unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"pinned checkpoint does not match native architecture: {loading}")
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise ValueError("quantized weights are prohibited")
    if sft_run:
        model = PeftModel.from_pretrained(model, sft_run / "model", is_trainable=True)
    else:
        # Restrict adapters to language attention/MLP projections. Text-only
        # examples cannot teach the vision tower or justify updating it.
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and "language_model" in name
                   and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}]
        if not targets:
            raise ValueError("native model exposes no expected language projection modules")
        model = get_peft_model(model, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=0.0,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        ))
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model


def sft_trainer_class():
    from transformers import Trainer

    from rocky.objectives import reply_logps

    class SpeechTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            sums, counts = reply_logps(model, inputs)
            # Each authored reply has equal weight, including very short replies.
            # Trainer handles gradient accumulation for this mean loss.
            loss = -(sums / counts).mean()
            return (loss, {"logits": sums.detach().unsqueeze(-1)}) if return_outputs else loss

    return SpeechTrainer


def encode_preferences(tokenizer, rows: list[dict], max_length: int) -> list[dict]:
    return [{key: encode_reply(tokenizer, row["prompt"], row[key][0]["content"], max_length)
             for key in ("chosen", "rejected")} for row in rows]


class PreferenceCollator:
    def __init__(self, pad_id):
        self.collate = ReplyCollator(pad_id)

    def __call__(self, rows):
        import torch

        batch = self.collate([r["chosen"] for r in rows] + [r["rejected"] for r in rows])
        if "reference_chosen" in rows[0]:
            batch["reference_chosen"] = torch.tensor([r["reference_chosen"] for r in rows])
            batch["reference_rejected"] = torch.tensor([r["reference_rejected"] for r in rows])
        return batch


def attach_reference(model, rows, pad_id, device):
    """Score the unchanged SFT model once, before any preference optimizer exists."""
    import torch

    from rocky.objectives import reply_logps

    collate = PreferenceCollator(pad_id)
    model.eval()
    with torch.no_grad():
        for row in rows:
            batch = {k: v.to(device) for k, v in collate([row]).items()}
            sums, _ = reply_logps(model, batch)
            row["reference_chosen"], row["reference_rejected"] = sums.cpu().tolist()


def preference_trainer_class():
    from transformers import Trainer

    from rocky.objectives import preference_loss, reply_logps

    class PreferenceTrainer(Trainer):
        def __init__(self, *args, objective, beta, gamma, **kwargs):
            super().__init__(*args, **kwargs)
            self.objective, self.beta, self.gamma = objective, beta, gamma
            self.model_accepts_loss_kwargs = False
            self.preference_metrics = None

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            self.preference_metrics = {"pairs": 0, "chosen_logp_per_token": 0.0,
                                       "rejected_logp_per_token": 0.0, "objective_win_rate": 0.0}
            try:
                result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
                totals = self.preference_metrics
                if totals["pairs"]:
                    metrics = {f"{metric_key_prefix}_{key}": value / totals["pairs"]
                               for key, value in totals.items() if key != "pairs"}
                    self.log(metrics)
                    result.update(metrics)
                return result
            finally:
                self.preference_metrics = None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            batch = dict(inputs)
            rc, rr = batch.pop("reference_chosen", None), batch.pop("reference_rejected", None)
            sums, counts = reply_logps(model, batch)
            half = len(sums) // 2
            losses = preference_loss(sums[:half], sums[half:], counts[:half], counts[half:],
                                     objective=self.objective, beta=self.beta, gamma=self.gamma,
                                     reference_chosen=rc, reference_rejected=rr)
            loss = losses.mean()
            if self.preference_metrics is not None:
                totals = self.preference_metrics
                totals["pairs"] += half
                totals["chosen_logp_per_token"] += (sums[:half] / counts[:half]).detach().sum().item()
                totals["rejected_logp_per_token"] += (sums[half:] / counts[half:]).detach().sum().item()
                # -logsigmoid(margin) < log(2) iff the objective margin is positive.
                totals["objective_win_rate"] += (losses.detach() < math.log(2)).sum().item()
            return (loss, {"logits": losses.detach().unsqueeze(-1)}) if return_outputs else loss

    return PreferenceTrainer


def run_training(*, config_path: Path = CONFIG, objective: str = "sft", curriculum: str = "core",
                 output: Path | None = None, preflight: bool = False, execute: bool = False,
                 resume: Path | None = None, sft_run: Path | None = None,
                 revision: str | None = None, hourly_usd: float | None = None,
                 wandb_mode: str = "offline", wandb_project: str = "rocky-final",
                 wandb_entity: str | None = None) -> dict:
    if objective not in OBJECTIVES:
        raise ValueError("unsupported objective; RL has no validated character reward here")
    config = load_config(config_path, require_model=preflight or execute, revision=revision)
    if curriculum not in {"core", "expanded"}:
        raise ValueError("unknown curriculum")
    if preflight and execute:
        raise ValueError("choose preflight or execution")
    if resume and not execute:
        raise ValueError("resume requires execution")
    if objective == "sft" and sft_run:
        raise ValueError("SFT starts from the pinned base; use --resume for interruption recovery")
    inventory = verify_export()
    train_name = ("core.train" if curriculum == "core" else "train") if objective == "sft" else "preferences.train"
    val_name = "validation" if objective == "sft" else "preferences.validation"
    train_rows = read_jsonl(DATA / f"exports/{train_name}.jsonl")
    validation_rows = read_jsonl(DATA / f"exports/{val_name}.jsonl")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ValueError("the $2/hour recipe is single-process; distributed launch is not supported")
    grad_accum = accumulation(config, world_size)
    if not preflight and not execute:
        return {"inventory": inventory, "objective": objective, "training_conversations_or_pairs": len(train_rows),
                "validation_conversations_or_pairs": len(validation_rows), "model_selected": config["model_id"] is not None,
                "training_started": False}
    if objective != "sft" and execute and sft_run is None:
        raise ValueError("preference execution requires --sft-run; never tune against an implicit base reference")
    check_tracking(wandb_mode, wandb_project)
    origin = sft_origin(sft_run, config) if sft_run else None
    from transformers import AutoTokenizer

    tokenizer_source, tokenizer_kwargs = tokenizer_location(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, **tokenizer_kwargs)
    if not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str):
        raise ValueError("a fast tokenizer with an explicit native template is required")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Gemma tokenizer has no padding token")
    if objective == "sft":
        train, train_stats = encode_conversations(tokenizer, train_rows, config["max_length"])
        validation, validation_stats = encode_conversations(tokenizer, validation_rows, config["max_length"])
    else:
        train = encode_preferences(tokenizer, train_rows, config["max_length"])
        validation = encode_preferences(tokenizer, validation_rows, config["max_length"])
        train_stats, validation_stats = {"pairs": len(train)}, {"pairs": len(validation)}
    checks = {"train": train_stats, "validation": validation_stats, "native_generation_prefix": True,
              "turn_termination_supervised": True, "tokens": {"train": token_diagnostics(train),
              "validation": token_diagnostics(validation)}, "wandb_mode": wandb_mode,
              "planned_optimizer_steps": math.ceil(len(train) / config["effective_batch_size"]) * config["epochs"]}
    # Validate test formatting without scoring it or using it for model selection.
    test_name = "test" if objective == "sft" else "preferences.test"
    test_rows = read_jsonl(DATA / f"exports/{test_name}.jsonl")
    test = (encode_conversations(tokenizer, test_rows, config["max_length"])[0] if objective == "sft"
            else encode_preferences(tokenizer, test_rows, config["max_length"]))
    checks["tokens"]["test_format_only"] = token_diagnostics(test)
    checks["provenance"] = {"model_id": config["model_id"], "revision": config["revision"],
                            "data_manifest_sha256": digest(DATA / "exports/manifest.json"),
                            "tokenizer_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                            "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                            "versions": {name: importlib.metadata.version(name) for name in
                                         ("torch", "transformers", "datasets", "accelerate", "peft")}}
    checks["gpu_execution_verified"] = False
    del test
    if not execute:
        return checks
    if output is None:
        raise ValueError("--output is required for execution")
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("execution requires CUDA with BF16 support; no precision/model fallback")
    if hourly_usd is None or not math.isfinite(hourly_usd) or not 0 < hourly_usd <= config["max_hourly_usd"]:
        raise ValueError("supply the actual --hourly-usd rate within the $2 budget")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total_gib < 44:
        raise ValueError("this unquantized E4B recipe requires a 48GB-class GPU; smaller-device fit is not validated")
    identity = {
        "objective": objective, "curriculum": curriculum, "config": config,
        "data_manifest": digest(DATA / "exports/manifest.json"), "training_data": digest(DATA / f"exports/{train_name}.jsonl"),
        "data_inputs": {name + ".jsonl": digest(DATA / f"exports/{name}.jsonl") for name in (train_name, val_name)},
        "tokenizer_template": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "tokenizer_backend": checks["provenance"]["tokenizer_backend_sha256"],
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "datasets", "accelerate", "peft")},
        "world_size": world_size, "gradient_accumulation": grad_accum,
        "sft_weights": origin["weights"] if origin else None,
    }
    rank = int(os.environ.get("RANK", "0"))
    record = prepare_run(output, identity, resume)
    record["preflight"] = checks
    record["status"] = "running"
    record["hourly_usd"] = hourly_usd
    write_json(output / "run.json", record)
    set_seed(config["seed"])
    started = time.monotonic()
    tracker = None
    torch.cuda.reset_peak_memory_stats()
    try:
        tracker = RunTracking(output, mode=wandb_mode, project=wandb_project, entity=wandb_entity,
                              identity=identity, diagnostics={"dataset": dataset_analysis(), "preflight": checks},
                              hourly_usd=hourly_usd)
        record["tracking"] = tracker.metadata
        if wandb_mode != "disabled":
            record["wandb_version"] = importlib.metadata.version("wandb")
        model = load_policy(config, sft_run)
        record["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        write_json(output / "run.json", record)
        args = TrainingArguments(
            output_dir=str(output), num_train_epochs=config["epochs"],
            learning_rate=config["learning_rate"] if objective == "sft" else config["preference_learning_rate"],
            per_device_train_batch_size=config["per_device_batch_size"], per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_ratio=config["warmup_ratio"], weight_decay=config["weight_decay"],
            lr_scheduler_type="cosine", max_grad_norm=1.0, optim="adamw_torch",
            eval_strategy="steps", save_strategy="steps", eval_steps=10, save_steps=10, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
            logging_steps=1, report_to="none", seed=config["seed"], data_seed=config["seed"],
            remove_unused_columns=False, prediction_loss_only=True,
        )
        if objective == "sft":
            trainer = sft_trainer_class()(model=model, args=args, train_dataset=Dataset.from_list(train),
                              eval_dataset=Dataset.from_list(validation), data_collator=ReplyCollator(tokenizer.pad_token_id))
        else:
            model.to("cuda")
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
            if objective in {"dpo", "dpo-norm"}:
                if resume:
                    if digest(output / "reference.json") != record["reference"]["sha256"]:
                        raise ValueError("reference scores changed after checkpoint")
                    references = json.loads((output / "reference.json").read_text())
                    for name, rows in (("train", train), ("validation", validation)):
                        if len(references[name]) != len(rows):
                            raise ValueError("reference row count changed")
                        for row, scores in zip(rows, references[name]):
                            row.update(scores)
                else:
                    attach_reference(model, train + validation, tokenizer.pad_token_id, "cuda")
                    references = {"train": [{k: r[k] for k in ("reference_chosen", "reference_rejected")} for r in train],
                                  "validation": [{k: r[k] for k in ("reference_chosen", "reference_rejected")} for r in validation]}
                    write_json(output / "reference.json", references)
                    record["reference"] = {"origin": "base plus unchanged SFT adapter before preference updates",
                                           "weights": origin["weights"], "sha256": digest(output / "reference.json")}
                write_json(output / "run.json", record)
            beta = config[{"dpo": "preference_beta", "dpo-norm": "normalized_dpo_beta", "simpo": "simpo_beta"}[objective]]
            trainer = preference_trainer_class()(model=model, args=args, train_dataset=Dataset.from_list(train),
                eval_dataset=Dataset.from_list(validation), data_collator=PreferenceCollator(tokenizer.pad_token_id),
                objective=objective, beta=beta, gamma=config["simpo_gamma"] if objective == "simpo" else 0.0)
        class FinalEvaluation(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= state.max_steps:
                    control.should_evaluate = True
                    control.should_save = True
                return control

        trainer.add_callback(FinalEvaluation())
        trainer.add_callback(tracking_callback(tracker))
        # Establish a validation baseline before updating; test data is never scored here.
        if not resume:
            record["baseline_metrics"] = trainer.evaluate(metric_key_prefix="eval_baseline")
        write_json(output / "run.json", record)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        # A tiny preference dataset can finish before eval_steps. Always evaluate
        # its final state and avoid presenting an absent checkpoint as 'best'.
        final_metrics = trainer.evaluate()
        if objective == "sft":
            manifest = json.loads((DATA / "exports/manifest.json").read_text())
            val_ids = manifest["row_ids"]["validation"]
            for name, prefix in (("film", "film-"), ("original", "original-"), ("expansion", None)):
                selected = [row for row_id, row in zip(val_ids, validation_rows)
                            if (row_id.startswith(prefix) if prefix else not row_id.startswith(("film-", "original-")))]
                if selected:
                    encoded, _ = encode_conversations(tokenizer, selected, config["max_length"])
                    final_metrics.update(trainer.evaluate(eval_dataset=Dataset.from_list(encoded),
                                                         metric_key_prefix="eval_" + name))
        trainer.save_model(str(output / "model"))
        tokenizer.save_pretrained(output / "model")
        record.update(status="completed", best_checkpoint=trainer.state.best_model_checkpoint,
                      selection="validation_loss_only", persona_accepted=False,
                      weights=weight_manifest(output / "model"), metrics=trainer.state.log_history)
        record.update(final_metrics=final_metrics, elapsed_seconds=time.monotonic() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        record["estimated_gpu_cost_usd"] = record["elapsed_seconds"] * hourly_usd / 3600
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__)
        record["elapsed_seconds"] = time.monotonic() - started
        record["estimated_gpu_cost_usd"] = record["elapsed_seconds"] * hourly_usd / 3600
        if rank == 0:
            write_json(output / "run.json", record)
        if tracker:
            try:
                tracker.finish(record, exit_code=1)
            except Exception:
                pass  # Keep the original training failure, already saved locally.
        raise
    write_json(output / "run.json", record)
    if tracker:
        try:
            tracker.finish(record)
        except Exception as error:
            record["tracking_finish_error"] = type(error).__name__
            write_json(output / "run.json", record)
    return record
