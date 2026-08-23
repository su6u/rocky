"""group hard-gate failures into reviewed data and engineering slices"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rocky_training.contracts_load import load_eval_gate_phrases
from rocky_training.eval_gates import (
    contains_any,
    has_single_response_object,
    response_is_valid,
)
from rocky_training.response_parse import parse_model_output
from rocky_training.trainer_jsonl import write_json


def inspect_result_failure(result: dict[str, Any]) -> dict[str, Any] | None:
    raw_output = str(result.get("rawOutput", ""))
    parsed = parse_model_output(raw_output)
    checks: list[str] = []

    if not has_single_response_object(raw_output):
        checks.append("response_single_object")
    if not response_is_valid(parsed.response_json):
        checks.append("response_schema_valid")
    phrases = load_eval_gate_phrases()
    if contains_any(parsed.spoken, phrases["bookFactTrapPhrases"]):
        checks.append("book_fact_trap")
    if contains_any(parsed.spoken, phrases["promptInjectionPhrases"]):
        checks.append("prompt_injection")

    if not checks:
        return None

    return {
        "promptId": result.get("promptId"),
        "scenarioFamily": result.get("scenarioFamily"),
        "checks": checks,
        "spoken": parsed.spoken,
        "rawOutput": raw_output,
        "suggestedAction": "route this failure family through the reviewed v1 data or engineering queue",
    }


def inspect_eval_failures(
    *,
    eval_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("eval payload must include results array")

    failures = [
        failure
        for result in results
        if isinstance(result, dict)
        for failure in [inspect_result_failure(result)]
        if failure is not None
    ]
    by_check: dict[str, int] = {}
    for failure in failures:
        for check in failure["checks"]:
            by_check[check] = by_check.get(check, 0) + 1

    report = {
        "sourceEvalPath": str(eval_path),
        "failureCount": len(failures),
        "byCheck": dict(sorted(by_check.items())),
        "failures": failures,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
