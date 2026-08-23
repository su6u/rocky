from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from rocky_training.model_spec import ModelSpec

LoadMode = Literal["train", "merge"]


class ModelLoadError(Exception):
    pass


@dataclass(frozen=True)
class LoadedTrainModel:
    model: Any
    processor: Any
    tokenizer: Any
    lora_config: Any | None


def resolve_lora_target_modules(base_model: str, spec: ModelSpec) -> list[str] | str:
    if "gemma-4" in base_model.lower():
        modules = "|".join(spec.adapter.target_modules)
        return rf".*language_model.*\.({modules})$"
    return list(spec.adapter.target_modules)


def resolve_lora_exclude_modules(base_model: str) -> list[str]:
    if "gemma-4" in base_model.lower():
        return ["vision_tower", "audio_tower", "multi_modal_projector"]
    return []


def _require_transformers() -> None:
    missing: list[str] = []
    for module in ("torch", "transformers", "peft"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise ModelLoadError(
            "missing train dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install -e 'training/.[train]'"
        )


def load_processor_and_tokenizer(base_model: str) -> tuple[Any, Any]:
    _require_transformers()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(base_model)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ModelLoadError("Gemma 4 processor did not expose a tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor, tokenizer


def build_lora_config(base_model: str, spec: ModelSpec) -> Any:
    _require_transformers()
    from peft import LoraConfig

    return LoraConfig(
        r=spec.adapter.rank,
        lora_alpha=spec.adapter.alpha,
        target_modules=resolve_lora_target_modules(base_model, spec),
        lora_dropout=spec.adapter.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        exclude_modules=resolve_lora_exclude_modules(base_model),
    )


def load_rocky_train_model(
    *,
    spec: ModelSpec,
    base_model: str,
    mode: LoadMode = "train",
    sft_adapter_dir: str | None = None,
) -> LoadedTrainModel:
    """Shared Gemma-4 multimodal load path for SFT/DPO/merge."""
    _require_transformers()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM

    processor, tokenizer = load_processor_and_tokenizer(base_model)
    compute_dtype = torch.bfloat16 if spec.train_precision == "bf16" else torch.float16

    if mode == "merge":
        model = AutoModelForMultimodalLM.from_pretrained(
            base_model,
            dtype=compute_dtype,
            device_map="auto",
        )
        return LoadedTrainModel(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            lora_config=None,
        )

    if spec.adapter.method == "qlora":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=spec.quantization.train,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            dtype=compute_dtype,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            base_model,
            dtype=compute_dtype,
            device_map="auto",
        )
    lora_config = build_lora_config(base_model, spec)

    if sft_adapter_dir is not None:
        model = PeftModel.from_pretrained(model, sft_adapter_dir, is_trainable=True)
        lora_config = None

    return LoadedTrainModel(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        lora_config=lora_config,
    )
