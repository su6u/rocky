from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rocky_training.checkpoint_select import CheckpointSelectError, select_best_from_eval_dir
from rocky_training.export_gguf import ExportGgufError, run_export_gguf
from rocky_training.inspect_failures import inspect_eval_failures
from rocky_training.interactive_eval import InteractiveEvalError, run_interactive_persona_eval
from rocky_training.merge_adapter import MergeAdapterError, run_merge_adapter
from rocky_training.model_spec import validate_model_spec_file
from rocky_training.paths import (
    default_persona_eval_path,
    default_persona_rubric_path,
    default_preference_dataset_path,
    default_spec_path,
    default_system_prompt_path,
)
from rocky_training.persona_rubric import PersonaRubricError
from rocky_training.run_eval import run_eval
from rocky_training.run_recipe import RecipeError, run_recipe
from rocky_training.smoke_sft import run_smoke_sft
from rocky_training.train_dpo import TrainDpoError, run_train_dpo
from rocky_training.train_sft import TrainSftError, run_train_sft

GPU_RUN_HELP = (
    "Full smoke and train runs expect a rented GPU. "
    "Local smoke may use CPU or a small GPU subset."
)


def _add_gpu_note(parser: argparse.ArgumentParser) -> None:
    parser.epilog = GPU_RUN_HELP


def _cmd_validate_spec(args: argparse.Namespace) -> int:
    result = validate_model_spec_file(args.spec)
    for issue in result.issues:
        print(f"{issue.path}: {issue.message}", file=sys.stderr)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if result.ok:
        print(f"ok: {result.file_path}")
        return 0
    return 1


def _cmd_smoke_sft(args: argparse.Namespace) -> int:
    manifest = run_smoke_sft(
        spec_path=Path(args.spec),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
        max_steps=args.max_steps,
        base_model=args.base_model,
    )
    print(json.dumps({"outputDir": args.output_dir, "manifest": manifest}, indent=2))
    return 0


def _cmd_run_eval(args: argparse.Namespace) -> int:
    payload = run_eval(
        host=args.host,
        model=args.model,
        output_path=Path(args.output),
        golden_path=Path(args.golden) if args.golden else None,
        spec_path=Path(args.spec),
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
        limit=args.limit,
        label=args.label,
        backend=args.backend,
        baseline_path=Path(args.baseline) if args.baseline else None,
        judge_models=tuple(args.judge_model or []),
        judge_host=args.judge_host,
        judge_backend=args.judge_backend,
        persona_rubric_path=Path(args.persona_rubric) if args.persona_rubric else None,
    )
    print(json.dumps({"output": args.output, "label": payload["label"], "count": len(payload["results"])}, indent=2))
    gate_failures = payload.get("gateSummary", {}).get("failures", [])
    if args.enforce_gates and gate_failures:
        print("eval gates failed:", file=sys.stderr)
        for failure in gate_failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


def _cmd_run_persona_eval(args: argparse.Namespace) -> int:
    args.golden = str(default_persona_eval_path())
    args.enforce_gates = True
    return _cmd_run_eval(args)


def _cmd_run_interactive_persona_eval(args: argparse.Namespace) -> int:
    payload = run_interactive_persona_eval(
        host=args.host,
        model=args.model,
        output_path=Path(args.output),
        trajectory_path=Path(args.trajectories) if args.trajectories else None,
        spec_path=Path(args.spec),
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
        limit=args.limit,
        label=args.label,
        backend=args.backend,
        judge_models=tuple(args.judge_model),
        judge_host=args.judge_host,
        judge_backend=args.judge_backend,
        persona_rubric_path=Path(args.persona_rubric),
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "label": payload["label"],
                "trajectories": payload["trajectoryCount"],
                "turns": payload["turnCount"],
            },
            indent=2,
        )
    )
    failures = payload.get("gateSummary", {}).get("failures", [])
    if failures:
        print("interactive persona gates failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


def _cmd_train_sft(args: argparse.Namespace) -> int:
    report_to = [] if args.report_to == "none" else [args.report_to]
    manifest = run_train_sft(
        spec_path=Path(args.spec),
        dataset_path=Path(args.dataset),
        validation_dataset_path=Path(args.validation_dataset) if args.validation_dataset else None,
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        max_rows=args.max_rows,
        max_validation_rows=args.max_validation_rows,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=report_to,
        dry_run=args.dry_run,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps({"outputDir": args.output_dir, "manifest": manifest}, indent=2))
    return 0


def _cmd_train_dpo(args: argparse.Namespace) -> int:
    report_to = [] if args.report_to == "none" else [args.report_to]
    manifest = run_train_dpo(
        spec_path=Path(args.spec),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        max_rows=args.max_rows,
        beta=args.beta,
        learning_rate=args.learning_rate,
        sft_adapter_dir=Path(args.sft_adapter_dir) if args.sft_adapter_dir else None,
        report_to=report_to,
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
        dry_run=args.dry_run,
    )
    print(json.dumps({"outputDir": args.output_dir, "manifest": manifest}, indent=2))
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    manifest = run_merge_adapter(
        spec_path=Path(args.spec),
        adapter_dir=Path(args.adapter_dir),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        dry_run=args.dry_run,
    )
    print(json.dumps({"outputDir": args.output_dir, "manifest": manifest}, indent=2))
    return 0


def _cmd_export_gguf(args: argparse.Namespace) -> int:
    manifest = run_export_gguf(
        spec_path=Path(args.spec),
        merged_dir=Path(args.merged_dir),
        output_path=Path(args.output_path),
        convert_script=Path(args.convert_script) if args.convert_script else None,
        quantize_binary=Path(args.quantize_binary) if args.quantize_binary else None,
        outtype=args.outtype,
        keep_intermediate=args.keep_intermediate,
        dry_run=args.dry_run,
    )
    print(json.dumps({"outputPath": args.output_path, "manifest": manifest}, indent=2))
    return 0


def _cmd_run_recipe(args: argparse.Namespace) -> int:
    lineage = run_recipe(
        spec_path=Path(args.spec),
        dataset_path=Path(args.dataset),
        run_id=args.run_id,
        include_dpo=args.include_dpo,
        preference_dataset_path=Path(args.preference_dataset) if args.preference_dataset else None,
        validation_dataset_path=Path(args.validation_dataset) if args.validation_dataset else None,
        dry_run=args.dry_run,
        base_model=args.base_model,
    )
    print(json.dumps(lineage, indent=2))
    return 0


def _cmd_inspect_eval_failures(args: argparse.Namespace) -> int:
    report = inspect_eval_failures(
        eval_path=Path(args.eval),
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(report, indent=2))
    return 1 if args.fail_on_findings and report["failureCount"] > 0 else 0


def _cmd_select_checkpoint(args: argparse.Namespace) -> int:
    report = select_best_from_eval_dir(Path(args.eval_dir), metric=args.metric)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocky_training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_spec = subparsers.add_parser("validate-spec", help="validate a ModelSpec yaml file")
    validate_spec.add_argument("--spec", type=str, default=str(default_spec_path()))
    validate_spec.set_defaults(handler=_cmd_validate_spec)

    smoke_sft = subparsers.add_parser("smoke-sft", help="run tiny adapter smoke training")
    _add_gpu_note(smoke_sft)
    smoke_sft.add_argument("--spec", type=str, default=str(default_spec_path()))
    smoke_sft.add_argument("--dataset", type=str, required=True)
    smoke_sft.add_argument("--output-dir", type=str, required=True)
    smoke_sft.add_argument("--max-rows", type=int, default=4)
    smoke_sft.add_argument("--max-steps", type=int, default=20)
    smoke_sft.add_argument("--base-model", type=str, default=None)
    smoke_sft.set_defaults(handler=_cmd_smoke_sft)

    train_sft = subparsers.add_parser("train-sft", help="run full sft training")
    _add_gpu_note(train_sft)
    train_sft.add_argument("--spec", type=str, default=str(default_spec_path()))
    train_sft.add_argument("--dataset", type=str, required=True)
    train_sft.add_argument("--validation-dataset", type=str, default=None)
    train_sft.add_argument("--output-dir", type=str, required=True)
    train_sft.add_argument("--base-model", type=str, default=None)
    train_sft.add_argument("--max-rows", type=int, default=0)
    train_sft.add_argument("--max-validation-rows", type=int, default=0)
    train_sft.add_argument("--per-device-train-batch-size", type=int, default=1)
    train_sft.add_argument("--per-device-eval-batch-size", type=int, default=1)
    train_sft.add_argument("--gradient-accumulation-steps", type=int, default=None)
    train_sft.add_argument("--report-to", type=str, default="none")
    train_sft.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="checkpoint path or 'latest' under output-dir/checkpoints",
    )
    train_sft.add_argument("--dry-run", action="store_true")
    train_sft.set_defaults(handler=_cmd_train_sft)

    train_dpo = subparsers.add_parser("train-dpo", help="run dpo training")
    _add_gpu_note(train_dpo)
    train_dpo.add_argument("--spec", type=str, default=str(default_spec_path()))
    train_dpo.add_argument("--dataset", type=str, default=str(default_preference_dataset_path()))
    train_dpo.add_argument("--output-dir", type=str, required=True)
    train_dpo.add_argument("--base-model", type=str, default=None)
    train_dpo.add_argument("--sft-adapter-dir", type=str, default=None)
    train_dpo.add_argument("--system-prompt", type=str, default=str(default_system_prompt_path()))
    train_dpo.add_argument("--max-rows", type=int, default=0)
    train_dpo.add_argument("--beta", type=float, default=0.1)
    train_dpo.add_argument("--learning-rate", type=float, default=5e-6)
    train_dpo.add_argument("--report-to", type=str, default="none")
    train_dpo.add_argument("--dry-run", action="store_true")
    train_dpo.set_defaults(handler=_cmd_train_dpo)

    merge = subparsers.add_parser("merge", help="merge adapter into base weights")
    _add_gpu_note(merge)
    merge.add_argument("--spec", type=str, default=str(default_spec_path()))
    merge.add_argument("--adapter-dir", type=str, required=True)
    merge.add_argument("--output-dir", type=str, required=True)
    merge.add_argument("--base-model", type=str, default=None)
    merge.add_argument("--dry-run", action="store_true")
    merge.set_defaults(handler=_cmd_merge)

    export_gguf = subparsers.add_parser("export-gguf", help="export merged weights to gguf")
    export_gguf.add_argument("--spec", type=str, default=str(default_spec_path()))
    export_gguf.add_argument("--merged-dir", type=str, required=True)
    export_gguf.add_argument("--output-path", type=str, required=True)
    export_gguf.add_argument("--convert-script", type=str, default=None)
    export_gguf.add_argument("--quantize-binary", type=str, default=None)
    export_gguf.add_argument("--outtype", type=str, default=None)
    export_gguf.add_argument("--keep-intermediate", action="store_true")
    export_gguf.add_argument("--dry-run", action="store_true")
    export_gguf.set_defaults(handler=_cmd_export_gguf)

    run_eval = subparsers.add_parser("run-eval", help="run baseline eval against a model endpoint")
    run_eval.add_argument("--host", type=str, default="http://localhost:11434")
    run_eval.add_argument("--model", type=str, required=True)
    run_eval.add_argument("--output", type=str, required=True)
    run_eval.add_argument("--golden", type=str, default=None)
    run_eval.add_argument("--spec", type=str, default=str(default_spec_path()))
    run_eval.add_argument("--system-prompt", type=str, default=str(default_system_prompt_path()))
    run_eval.add_argument("--limit", type=int, default=0)
    run_eval.add_argument("--label", type=str, default=None)
    run_eval.add_argument("--backend", type=str, choices=["ollama", "llama-cpp"], default="ollama")
    run_eval.add_argument("--baseline", type=str, default=None)
    run_eval.add_argument("--judge-model", action="append", default=[])
    run_eval.add_argument("--judge-host", type=str, default=None)
    run_eval.add_argument(
        "--judge-backend",
        type=str,
        choices=["ollama", "llama-cpp"],
        default="ollama",
    )
    run_eval.add_argument(
        "--persona-rubric",
        type=str,
        default=str(default_persona_rubric_path()),
    )
    run_eval.add_argument("--enforce-gates", action="store_true")
    run_eval.set_defaults(handler=_cmd_run_eval)

    persona_eval = subparsers.add_parser(
        "run-persona-eval",
        help="run 100-prompt arbitrary-topic Rocky persona gate",
    )
    persona_eval.add_argument("--host", type=str, default="http://localhost:11434")
    persona_eval.add_argument("--model", type=str, required=True)
    persona_eval.add_argument("--output", type=str, required=True)
    persona_eval.add_argument("--spec", type=str, default=str(default_spec_path()))
    persona_eval.add_argument("--system-prompt", type=str, default=str(default_system_prompt_path()))
    persona_eval.add_argument("--limit", type=int, default=0)
    persona_eval.add_argument("--label", type=str, default=None)
    persona_eval.add_argument("--backend", type=str, choices=["ollama", "llama-cpp"], default="ollama")
    persona_eval.add_argument("--baseline", type=str, default=None)
    persona_eval.add_argument("--judge-model", action="append", required=True)
    persona_eval.add_argument("--judge-host", type=str, default=None)
    persona_eval.add_argument(
        "--judge-backend",
        type=str,
        choices=["ollama", "llama-cpp"],
        default="ollama",
    )
    persona_eval.add_argument(
        "--persona-rubric",
        type=str,
        default=str(default_persona_rubric_path()),
    )
    persona_eval.set_defaults(handler=_cmd_run_persona_eval)

    interactive_eval = subparsers.add_parser(
        "run-interactive-persona-eval",
        help="run live multi-turn Rocky persona trajectories",
    )
    interactive_eval.add_argument("--host", type=str, default="http://localhost:11434")
    interactive_eval.add_argument("--model", type=str, required=True)
    interactive_eval.add_argument("--output", type=str, required=True)
    interactive_eval.add_argument("--trajectories", type=str, default=None)
    interactive_eval.add_argument("--spec", type=str, default=str(default_spec_path()))
    interactive_eval.add_argument(
        "--system-prompt",
        type=str,
        default=str(default_system_prompt_path()),
    )
    interactive_eval.add_argument("--limit", type=int, default=0)
    interactive_eval.add_argument("--label", type=str, default=None)
    interactive_eval.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "llama-cpp"],
        default="ollama",
    )
    interactive_eval.add_argument("--judge-model", action="append", required=True)
    interactive_eval.add_argument("--judge-host", type=str, default=None)
    interactive_eval.add_argument(
        "--judge-backend",
        type=str,
        choices=["ollama", "llama-cpp"],
        default="ollama",
    )
    interactive_eval.add_argument(
        "--persona-rubric",
        type=str,
        default=str(default_persona_rubric_path()),
    )
    interactive_eval.set_defaults(handler=_cmd_run_interactive_persona_eval)

    recipe = subparsers.add_parser(
        "run-recipe",
        help="orchestrate sft → optional dpo → merge → gguf from ModelSpec artifacts",
    )
    _add_gpu_note(recipe)
    recipe.add_argument("--spec", type=str, default=str(default_spec_path()))
    recipe.add_argument("--dataset", type=str, required=True)
    recipe.add_argument("--validation-dataset", type=str, default=None)
    recipe.add_argument("--preference-dataset", type=str, default=None)
    recipe.add_argument("--run-id", type=str, default=None)
    recipe.add_argument("--include-dpo", action="store_true")
    recipe.add_argument("--base-model", type=str, default=None)
    recipe.add_argument("--dry-run", action="store_true")
    recipe.set_defaults(handler=_cmd_run_recipe)

    inspect_failures = subparsers.add_parser(
        "inspect-eval-failures",
        help="group deterministic eval failures for targeted corpus fixes",
    )
    inspect_failures.add_argument("--eval", type=str, required=True)
    inspect_failures.add_argument("--output", type=str, default=None)
    inspect_failures.add_argument("--fail-on-findings", action="store_true")
    inspect_failures.set_defaults(handler=_cmd_inspect_eval_failures)

    select_checkpoint = subparsers.add_parser(
        "select-checkpoint",
        help="pick best checkpoint from eval payload dir (eval_loss or composite_gates)",
    )
    select_checkpoint.add_argument("--eval-dir", type=str, required=True)
    select_checkpoint.add_argument(
        "--metric",
        type=str,
        default="composite_gates",
        choices=["eval_loss", "composite_gates"],
    )
    select_checkpoint.add_argument("--output", type=str, default=None)
    select_checkpoint.set_defaults(handler=_cmd_select_checkpoint)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        TrainSftError,
        TrainDpoError,
        MergeAdapterError,
        ExportGgufError,
        RecipeError,
        CheckpointSelectError,
        InteractiveEvalError,
        PersonaRubricError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
