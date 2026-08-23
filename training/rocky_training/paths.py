"""centralize v1 repository paths used by training commands"""

from pathlib import Path

DEFAULT_GEMMA_E4B_IT = "google/gemma-4-E4B-it"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def training_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_spec_path() -> Path:
    return repo_root() / "configs" / "rocky-v1.yaml"


def default_contracts_dir() -> Path:
    return repo_root() / "contracts"


def default_data_release_manifest_path() -> Path:
    return repo_root() / "data" / "v1" / "release-manifest.json"


def default_response_schema_path() -> Path:
    return default_contracts_dir() / "v1" / "rocky-response.schema.json"


def default_golden_eval_path() -> Path:
    return repo_root() / "data" / "v1" / "eval" / "golden.jsonl"


def default_persona_eval_path() -> Path:
    return repo_root() / "data" / "v1" / "eval" / "persona-holdout.jsonl"


def default_interactive_persona_eval_path() -> Path:
    return (
        repo_root()
        / "data"
        / "v1"
        / "eval"
        / "persona-interactive.jsonl"
    )


def default_persona_rubric_path() -> Path:
    return repo_root() / "contracts" / "v1" / "persona-rubric.json"


def default_preference_dataset_path() -> Path:
    return repo_root() / "data" / "v1" / "exports" / "preferences.jsonl"


def default_system_prompt_path() -> Path:
    return default_contracts_dir() / "prompts" / "rocky-system.txt"
