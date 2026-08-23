from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from rocky_training.model_spec import ModelSpec, load_model_spec
from rocky_training.train_sft import resolve_train_base_model
from rocky_training.trainer_jsonl import write_json


class MergeAdapterError(Exception):
    pass


@dataclass(frozen=True)
class MergeAdapterResult:
    base_model: str
    adapter_dir: str
    merged_dir: str


def _require_merge_dependencies() -> None:
    missing: list[str] = []
    for module in ("peft", "torch", "transformers"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise MergeAdapterError(
            "missing merge dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install -e 'training/.[train]'"
        )


def merge_adapter_into_base(
    *,
    spec: ModelSpec,
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
) -> MergeAdapterResult:
    if not adapter_dir.is_dir():
        raise MergeAdapterError(f"adapter directory not found: {adapter_dir}")

    _require_merge_dependencies()
    from peft import PeftModel

    from rocky_training.model_load import ModelLoadError, load_rocky_train_model

    try:
        loaded = load_rocky_train_model(spec=spec, base_model=base_model, mode="merge")
    except ModelLoadError as error:
        raise MergeAdapterError(str(error)) from error

    model = PeftModel.from_pretrained(loaded.model, str(adapter_dir))
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    loaded.processor.save_pretrained(output_dir)

    return MergeAdapterResult(
        base_model=base_model,
        adapter_dir=str(adapter_dir),
        merged_dir=str(output_dir),
    )


def build_merge_manifest(
    *,
    spec: ModelSpec,
    adapter_dir: Path,
    output_dir: Path,
    base_model: str,
    merge: MergeAdapterResult | None,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "kind": "merge-adapter",
        "dryRun": dry_run,
        "specId": spec.id,
        "baseModel": base_model,
        "trainPrecision": spec.train_precision,
        "adapterDir": str(adapter_dir),
        "mergedDir": merge.merged_dir if merge else str(output_dir),
        "finishedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


def run_merge_adapter(
    *,
    spec_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    base_model: str | None = None,
    dry_run: bool = False,
    merge_runner: Callable[..., MergeAdapterResult] | None = None,
) -> dict[str, Any]:
    spec = load_model_spec(spec_path)
    resolved_base_model = resolve_train_base_model(spec, base_model)

    merge: MergeAdapterResult | None = None
    if not dry_run:
        runner = merge_runner or merge_adapter_into_base
        merge = runner(
            spec=spec,
            base_model=resolved_base_model,
            adapter_dir=adapter_dir,
            output_dir=output_dir,
        )

    manifest = build_merge_manifest(
        spec=spec,
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        base_model=resolved_base_model,
        merge=merge,
        dry_run=dry_run,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest
