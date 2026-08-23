"""validate the checked-in v1 model specification"""

import subprocess
import sys

from rocky_training.model_spec import (
    MODEL_SPEC_BASE_MODEL_WARNING,
    MODEL_SPEC_EXPORT_QUANT_WARNING,
    validate_model_spec,
    validate_model_spec_file,
)
from rocky_training.paths import default_spec_path, repo_root


def test_validate_model_spec_file_accepts_checked_in_spec() -> None:
    result = validate_model_spec_file(default_spec_path())
    assert result.ok, [f"{issue.path}: {issue.message}" for issue in result.issues]
    assert result.spec is not None
    assert result.spec.id == "rocky-v1"
    assert result.spec.base_model == "google/gemma-4-31B-it"
    assert result.spec.chat_template == "gemma4"
    assert result.spec.enable_thinking is False
    assert result.spec.checkpoint_metric == "composite_gates"
    assert result.spec.adapter.dropout == 0.05
    assert result.spec.optimizer.warmup_ratio == 0.05
    assert result.spec.optimizer.warmup_steps == 0
    assert result.spec.eval_gates.rocky_persona_rate == 0.95
    assert MODEL_SPEC_EXPORT_QUANT_WARNING not in result.warnings
    assert MODEL_SPEC_BASE_MODEL_WARNING in result.warnings
    assert not any("PLACEHOLDER_" in warning for warning in result.warnings)


def test_validate_model_spec_rejects_invalid_adapter_rank() -> None:
    result = validate_model_spec(
        {
            "id": "bad",
            "base_model": "org/model",
            "base_model_fallback": "org/fallback",
            "chat_template": "gemma",
            "enable_thinking": False,
            "checkpoint_metric": "eval_loss",
            "train_precision": "bf16",
            "quantization": {"train": "nf4", "export": "q4_k_m"},
            "sequence": {"max_length": 4096},
            "adapter": {
                "method": "qlora",
                "rank": 0,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj"],
            },
            "optimizer": {
                "learning_rate": 0.0001,
                "scheduler": "cosine",
                "warmup_steps": 10,
                "weight_decay": 0.01,
                "effective_batch_size": 16,
                "max_epochs": 3,
                "early_stopping": True,
            },
            "inference": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096,
                "stop": ["<turn|>"],
            },
            "artifacts": {
                "adapter_dir": "a",
                "merged_dir": "m",
                "gguf_path": "g",
                "modelfile_path": "f",
            },
            "eval_gates": {
                "response_schema_valid_rate": 0.98,
                "response_single_object_rate": 0.98,
                "book_fact_contradiction_rate": 0.02,
                "prompt_injection_fail_rate": 0.05,
                "rocky_persona_rate": 0.9,
            },
        }
    )
    assert any(issue.path == "adapter.rank" for issue in result.issues)


def test_validate_model_spec_rejects_fractional_adapter_rank() -> None:
    result = validate_model_spec(
        {
            "id": "bad",
            "base_model": "org/model",
            "base_model_fallback": "org/fallback",
            "chat_template": "gemma",
            "enable_thinking": False,
            "checkpoint_metric": "eval_loss",
            "train_precision": "bf16",
            "quantization": {"train": "nf4", "export": "q4_k_m"},
            "sequence": {"max_length": 4096},
            "adapter": {
                "method": "qlora",
                "rank": 0.9,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj"],
            },
            "optimizer": {
                "learning_rate": 0.0001,
                "scheduler": "cosine",
                "warmup_steps": 10,
                "weight_decay": 0.01,
                "effective_batch_size": 16,
                "max_epochs": 3,
                "early_stopping": True,
            },
            "inference": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096,
                "stop": ["<turn|>"],
            },
            "artifacts": {
                "adapter_dir": "a",
                "merged_dir": "m",
                "gguf_path": "g",
                "modelfile_path": "f",
            },
            "eval_gates": {
                "response_schema_valid_rate": 0.98,
                "response_single_object_rate": 0.98,
                "book_fact_contradiction_rate": 0.02,
                "prompt_injection_fail_rate": 0.05,
                "rocky_persona_rate": 0.9,
            },
        }
    )
    assert any(issue.path == "adapter.rank" for issue in result.issues)


def test_module_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "rocky_training", "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root()),
    )
    assert "validate-spec" in completed.stdout
    assert "smoke-sft" in completed.stdout
    assert "run-eval" in completed.stdout
