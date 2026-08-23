import json
from pathlib import Path

import pytest

from rocky_training.merge_adapter import (
    MergeAdapterError,
    MergeAdapterResult,
    merge_adapter_into_base,
    run_merge_adapter,
)
from rocky_training.paths import default_spec_path


def test_run_merge_adapter_dry_run_writes_manifest(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    output_dir = tmp_path / "merged"

    manifest = run_merge_adapter(
        spec_path=default_spec_path(),
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        dry_run=True,
    )

    assert manifest["kind"] == "merge-adapter"
    assert manifest["dryRun"] is True
    assert manifest["mergedDir"] == str(output_dir)
    saved = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["adapterDir"] == str(adapter_dir)


def test_run_merge_adapter_fake_runner(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    output_dir = tmp_path / "merged"

    def fake_runner(**kwargs: object) -> MergeAdapterResult:
        assert kwargs["adapter_dir"] == adapter_dir
        return MergeAdapterResult(
            base_model="google/gemma-4-E4B-it",
            adapter_dir=str(adapter_dir),
            merged_dir=str(output_dir),
        )

    manifest = run_merge_adapter(
        spec_path=default_spec_path(),
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        merge_runner=fake_runner,
    )

    assert manifest["dryRun"] is False
    assert manifest["mergedDir"] == str(output_dir)


def test_merge_adapter_into_base_requires_adapter_dir(tmp_path: Path) -> None:
    from rocky_training.model_spec import load_model_spec

    spec = load_model_spec(default_spec_path())
    with pytest.raises(MergeAdapterError, match="adapter directory not found"):
        merge_adapter_into_base(
            spec=spec,
            base_model="google/gemma-4-E4B-it",
            adapter_dir=tmp_path / "missing",
            output_dir=tmp_path / "merged",
        )
