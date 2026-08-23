"""verify candidate export lineage for bf16 and explicit quantization"""

import json
from pathlib import Path

import pytest

from rocky_training.export_gguf import (
    ExportGgufError,
    ExportGgufResult,
    export_merged_to_gguf,
    resolve_convert_script,
    run_export_gguf,
)
from rocky_training.paths import default_spec_path


def test_run_export_gguf_dry_run_writes_manifest(tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "config.json").write_text("{}", encoding="utf-8")
    output_path = tmp_path / "rocky-bf16.gguf"

    manifest = run_export_gguf(
        spec_path=default_spec_path(),
        merged_dir=merged_dir,
        output_path=output_path,
        dry_run=True,
    )

    assert manifest["kind"] == "export-gguf"
    assert manifest["dryRun"] is True
    assert manifest["outtype"] == "bf16"
    saved = json.loads((tmp_path / "export-gguf.manifest.json").read_text(encoding="utf-8"))
    assert saved["outputPath"] == str(output_path)


def test_run_export_gguf_fake_runner(tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "config.json").write_text("{}", encoding="utf-8")
    output_path = tmp_path / "rocky-q4_k_m.gguf"
    convert_script = tmp_path / "convert.py"
    quantize_binary = tmp_path / "llama-quantize"
    convert_script.write_text("print('ok')", encoding="utf-8")
    quantize_binary.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_runner(**kwargs: object) -> ExportGgufResult:
        assert kwargs["merged_dir"] == merged_dir
        return ExportGgufResult(
            merged_dir=str(merged_dir),
            output_path=str(output_path),
            outtype="q4_k_m",
            convert_script=str(convert_script),
            quantize_binary=str(quantize_binary),
        )

    manifest = run_export_gguf(
        spec_path=default_spec_path(),
        merged_dir=merged_dir,
        output_path=output_path,
        convert_script=convert_script,
        quantize_binary=quantize_binary,
        export_runner=fake_runner,
    )

    assert manifest["dryRun"] is False
    assert manifest["convertScript"] == str(convert_script)
    assert manifest["quantizeBinary"] == str(quantize_binary)


def test_resolve_convert_script_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(ExportGgufError, match="convert script not found"):
        resolve_convert_script(tmp_path / "missing.py")


def test_export_merged_to_gguf_requires_config_json(tmp_path: Path) -> None:
    from rocky_training.model_spec import load_model_spec

    spec = load_model_spec(default_spec_path())
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    convert_script = tmp_path / "convert.py"
    convert_script.write_text("print('ok')", encoding="utf-8")

    with pytest.raises(ExportGgufError, match="missing config.json"):
        export_merged_to_gguf(
            spec=spec,
            merged_dir=merged_dir,
            output_path=tmp_path / "out.gguf",
            convert_script=convert_script,
        )
