"""build candidate artifacts without implying production promotion"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rocky_training.export_gguf import run_export_gguf
from rocky_training.merge_adapter import run_merge_adapter
from rocky_training.model_spec import load_model_spec
from rocky_training.paths import default_preference_dataset_path, repo_root
from rocky_training.trainer_jsonl import write_json
from rocky_training.train_dpo import run_train_dpo
from rocky_training.train_sft import run_train_sft


class RecipeError(Exception):
    pass


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def run_recipe(
    *,
    spec_path: Path,
    dataset_path: Path,
    run_id: str | None = None,
    include_dpo: bool = False,
    preference_dataset_path: Path | None = None,
    validation_dataset_path: Path | None = None,
    dry_run: bool = False,
    base_model: str | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """orchestrate SFT, optional preference tuning, merge, and candidate export"""
    spec = load_model_spec(spec_path)
    stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (run_root or repo_root() / "runs") / f"{spec.id}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = _resolve_repo_path(spec.artifacts.adapter_dir)
    merged_dir = _resolve_repo_path(spec.artifacts.merged_dir)
    gguf_path = _resolve_repo_path(spec.artifacts.gguf_path)

    lineage: dict[str, Any] = {
        "kind": "recipe",
        "specId": spec.id,
        "runId": stamp,
        "runDir": str(run_dir),
        "datasetPath": str(dataset_path),
        "stages": {},
        "dryRun": dry_run,
        "promotionStatus": "candidate_only",
        "releaseEligible": False,
        "requiredNextStage": "bf16 and serving-artifact evaluation with fail-closed gates",
    }

    sft_manifest = run_train_sft(
        spec_path=spec_path,
        dataset_path=dataset_path,
        validation_dataset_path=validation_dataset_path,
        output_dir=adapter_dir,
        base_model=base_model,
        dry_run=dry_run,
    )
    lineage["stages"]["train_sft"] = sft_manifest

    active_adapter = adapter_dir
    if include_dpo:
        dpo_dir = run_dir / "dpo-adapter"
        dpo_manifest = run_train_dpo(
            spec_path=spec_path,
            dataset_path=preference_dataset_path or default_preference_dataset_path(),
            output_dir=dpo_dir,
            base_model=base_model,
            sft_adapter_dir=adapter_dir,
            dry_run=dry_run,
        )
        lineage["stages"]["train_dpo"] = dpo_manifest
        active_adapter = dpo_dir

    merge_manifest = run_merge_adapter(
        spec_path=spec_path,
        adapter_dir=active_adapter,
        output_dir=merged_dir,
        base_model=base_model,
        dry_run=dry_run,
    )
    lineage["stages"]["merge"] = merge_manifest

    export_manifest = run_export_gguf(
        spec_path=spec_path,
        merged_dir=merged_dir,
        output_path=gguf_path,
        dry_run=dry_run,
    )
    lineage["stages"]["export_gguf"] = export_manifest
    lineage["artifacts"] = {
        "adapterDir": str(active_adapter),
        "mergedDir": str(merged_dir),
        "ggufPath": str(gguf_path),
        "modelfilePath": str(_resolve_repo_path(spec.artifacts.modelfile_path)),
    }

    write_json(run_dir / "lineage.json", lineage)
    return lineage
