from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from rocky_training.model_spec import ModelSpec, load_model_spec
from rocky_training.trainer_jsonl import write_json


class ExportGgufError(Exception):
    pass


@dataclass(frozen=True)
class ExportGgufResult:
    merged_dir: str
    output_path: str
    outtype: str
    convert_script: str
    quantize_binary: str | None = None
    intermediate_path: str | None = None


def resolve_convert_script(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise ExportGgufError(f"convert script not found: {explicit}")
        return explicit

    env_path = __import__("os").environ.get("LLAMA_CPP_CONVERT_SCRIPT")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path

    raise ExportGgufError(
        "llama.cpp convert script required; pass --convert-script or set LLAMA_CPP_CONVERT_SCRIPT"
    )


def resolve_quantize_binary(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise ExportGgufError(f"quantize binary not found: {explicit}")
        return explicit

    env_path = __import__("os").environ.get("LLAMA_CPP_QUANTIZE_BINARY")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path

    raise ExportGgufError(
        "llama.cpp quantize binary required for q4_k_m; pass --quantize-binary or set LLAMA_CPP_QUANTIZE_BINARY"
    )


def run_checked(command: list[str], failure_label: str) -> None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise ExportGgufError(f"{failure_label} failed: {stderr}")


def export_merged_to_gguf(
    *,
    spec: ModelSpec,
    merged_dir: Path,
    output_path: Path,
    convert_script: Path,
    quantize_binary: Path | None = None,
    outtype: str | None = None,
    keep_intermediate: bool = False,
) -> ExportGgufResult:
    if not merged_dir.is_dir():
        raise ExportGgufError(f"merged model directory not found: {merged_dir}")
    if not (merged_dir / "config.json").is_file():
        raise ExportGgufError(f"merged model missing config.json: {merged_dir}")

    resolved_outtype = outtype or spec.quantization.export
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_outtype == "q4_k_m":
        resolved_quantize_binary = resolve_quantize_binary(quantize_binary)
        intermediate_path = output_path.with_name(f"{output_path.stem}.bf16.gguf")
        run_checked(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile",
                str(intermediate_path),
                "--outtype",
                "bf16",
            ],
            "gguf bf16 export",
        )
        run_checked(
            [
                str(resolved_quantize_binary),
                str(intermediate_path),
                str(output_path),
                "Q4_K_M",
            ],
            "gguf quantize",
        )
        if not keep_intermediate:
            intermediate_path.unlink(missing_ok=True)
        if not output_path.is_file():
            raise ExportGgufError(f"gguf export did not create output file: {output_path}")
        return ExportGgufResult(
            merged_dir=str(merged_dir),
            output_path=str(output_path),
            outtype=resolved_outtype,
            convert_script=str(convert_script),
            quantize_binary=str(resolved_quantize_binary),
            intermediate_path=str(intermediate_path),
        )

    run_checked(
        [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile",
            str(output_path),
            "--outtype",
            resolved_outtype,
        ],
        "gguf export",
    )

    if not output_path.is_file():
        raise ExportGgufError(f"gguf export did not create output file: {output_path}")

    return ExportGgufResult(
        merged_dir=str(merged_dir),
        output_path=str(output_path),
        outtype=resolved_outtype,
        convert_script=str(convert_script),
    )


def build_export_gguf_manifest(
    *,
    spec: ModelSpec,
    merged_dir: Path,
    output_path: Path,
    export: ExportGgufResult | None,
    dry_run: bool,
    convert_script: Path | None,
    quantize_binary: Path | None,
) -> dict[str, Any]:
    return {
        "kind": "export-gguf",
        "dryRun": dry_run,
        "specId": spec.id,
        "mergedDir": str(merged_dir),
        "outputPath": str(output_path),
        "outtype": export.outtype if export else spec.quantization.export,
        "convertScript": export.convert_script if export else (str(convert_script) if convert_script else None),
        "quantizeBinary": export.quantize_binary if export else (str(quantize_binary) if quantize_binary else None),
        "intermediatePath": export.intermediate_path if export else None,
        "modelfilePath": spec.artifacts.modelfile_path,
        "finishedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


def run_export_gguf(
    *,
    spec_path: Path,
    merged_dir: Path,
    output_path: Path,
    convert_script: Path | None = None,
    quantize_binary: Path | None = None,
    outtype: str | None = None,
    keep_intermediate: bool = False,
    dry_run: bool = False,
    export_runner: Callable[..., ExportGgufResult] | None = None,
) -> dict[str, Any]:
    spec = load_model_spec(spec_path)
    resolved_convert_script = None if dry_run else resolve_convert_script(convert_script)
    resolved_outtype = outtype or spec.quantization.export
    resolved_quantize_binary = (
        None if dry_run or resolved_outtype != "q4_k_m" else resolve_quantize_binary(quantize_binary)
    )

    export: ExportGgufResult | None = None
    if not dry_run:
        if resolved_convert_script is None:
            raise ExportGgufError("convert script is required for export")
        runner = export_runner or export_merged_to_gguf
        export = runner(
            spec=spec,
            merged_dir=merged_dir,
            output_path=output_path,
            convert_script=resolved_convert_script,
            quantize_binary=resolved_quantize_binary,
            outtype=outtype,
            keep_intermediate=keep_intermediate,
        )

    manifest = build_export_gguf_manifest(
        spec=spec,
        merged_dir=merged_dir,
        output_path=output_path,
        export=export,
        dry_run=dry_run,
        convert_script=resolved_convert_script,
        quantize_binary=resolved_quantize_binary,
    )
    write_json(output_path.parent / "export-gguf.manifest.json", manifest)
    return manifest
