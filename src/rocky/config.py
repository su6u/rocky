"""One explicit Gemma 4 configuration. Never fall back or quantize silently."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from rocky.data import ROOT

CONFIG = ROOT / "configs/training.json"
LOCAL_TOKENIZER = ROOT / ".cache/gemma-e4b-tokenizer"
LOCAL_TOKENIZER_RECORD = ROOT / ".cache/e4b-tokenizer-check.json"
# E4B means effective size; its per-layer embeddings also remain loaded.
# The larger A4B MoE, quantized variants, and fallback model IDs are invalid.
MODELS = {"google/gemma-4-E4B-it"}
FIELDS = {"model_id", "revision", "precision", "quantization", "update", "max_length", "epochs", "learning_rate", "preference_learning_rate", "effective_batch_size", "per_device_batch_size", "warmup_ratio", "weight_decay", "seed", "preference_beta", "simpo_beta", "simpo_gamma", "normalized_dpo_beta", "lora_rank", "lora_alpha", "max_hourly_usd"}


def load_config(path: Path = CONFIG, *, require_model: bool = True, revision: str | None = None) -> dict:
    config = json.loads(path.read_text())
    if revision is not None:
        config["revision"] = revision
    if set(config) != FIELDS:
        raise ValueError("training config has missing or unknown fields")
    if config["precision"] != "bf16" or config["quantization"] != "none" or config["update"] != "lora":
        raise ValueError("the budgeted recipe requires unquantized BF16 Gemma 4 with LoRA updates")
    if require_model or config["model_id"] is not None:
        if config["model_id"] not in MODELS:
            raise ValueError("select the exact official Gemma 4 instruction-tuned model; no fallback")
    if require_model or config["revision"] is not None:
        if not isinstance(config["revision"], str) or not re.fullmatch(r"[a-f0-9]{40}", config["revision"]):
            raise ValueError("pin the official model to its exact 40-character commit")
    for key in ("max_length", "epochs", "effective_batch_size", "per_device_batch_size", "lora_rank", "lora_alpha"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_length"] < 128 or type(config["seed"]) is not int:
        raise ValueError("invalid sequence length, epoch count, or seed")
    for key in ("learning_rate", "preference_learning_rate", "preference_beta", "simpo_beta", "normalized_dpo_beta", "simpo_gamma", "warmup_ratio", "weight_decay", "max_hourly_usd"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid finite parameter: {key}")
    if not 0 < config["learning_rate"] <= 1e-3 or not 0 < config["preference_learning_rate"] <= 1e-3:
        raise ValueError("invalid learning rate")
    if config["warmup_ratio"] > 1 or any(config[k] <= 0 for k in ("preference_beta", "simpo_beta", "normalized_dpo_beta")):
        raise ValueError("invalid warmup or preference beta")
    if not 0 < config["max_hourly_usd"] <= 2:
        raise ValueError("hourly GPU budget must remain within $2")
    return config


def accumulation(config: dict, world_size: int) -> int:
    divisor = config["per_device_batch_size"] * world_size
    if world_size < 1 or config["effective_batch_size"] % divisor or config["effective_batch_size"] < divisor:
        raise ValueError("effective batch must be divisible by per-device batch × GPU processes")
    return config["effective_batch_size"] // divisor


def tokenizer_location(config: dict) -> tuple[str, dict]:
    """Prefer the locally audited tokenizer; otherwise use the exact remote revision."""
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    if LOCAL_TOKENIZER_RECORD.is_file() and all((LOCAL_TOKENIZER / name).is_file() for name in required):
        record = json.loads(LOCAL_TOKENIZER_RECORD.read_text())
        if record.get("model") == config["model_id"] and record.get("revision") == config["revision"]:
            return str(LOCAL_TOKENIZER), {"local_files_only": True}
    return config["model_id"], {"revision": config["revision"]}
