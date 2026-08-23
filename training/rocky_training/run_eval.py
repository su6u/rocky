"""run constrained endpoint evaluation and write a reproducible report"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from rocky_training.endpoint_client import (
    call_llama_cpp_chat,
    call_ollama_chat,
    load_rocky_response_schema,
)
from rocky_training.eval_gates import evaluate_gate_summary, serialize_gate_summary
from rocky_training.golden_prompts import GoldenPrompt, load_golden_prompts
from rocky_training.response_parse import parse_model_output, slugify_label
from rocky_training.model_spec import load_model_spec
from rocky_training.paths import (
    default_golden_eval_path,
    default_persona_rubric_path,
    default_spec_path,
    default_system_prompt_path,
)
from rocky_training.persona_rubric import judge_persona_results, load_persona_rubric
from rocky_training.trainer_jsonl import write_json
from rocky_training.turn_context import prepare_turn_messages


@dataclass(frozen=True)
class EvalResultRow:
    id: str
    prompt_id: str
    scenario_family: str
    user: str
    raw_output: str
    spoken: str
    response_json: str | None
    quality_focus: str | None = None
    humor_expectation: str | None = None
    grounding_patterns: tuple[str, ...] = ()
    uncertainty_patterns: tuple[str, ...] = ()
    roleplay_forbidden_patterns: tuple[str, ...] = ()
    book_fact_forbidden_patterns: tuple[str, ...] = ()


ChatCaller = Callable[..., str]


def load_system_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path is not None else default_system_prompt_path()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"system prompt file not found: {prompt_path}")
    # Match @rocky/prompt SYSTEM_PROMPT (sync-contracts adds a trailing newline).
    return prompt_path.read_text(encoding="utf-8").rstrip("\n")


def build_eval_messages(
    system_prompt: str,
    user_prompt: str,
    *,
    grounding_notes: str | None = None,
) -> list[dict[str, str]]:
    turn = prepare_turn_messages(
        [{"role": "user", "content": user_prompt}],
        grounding_notes=grounding_notes,
    )
    return [{"role": "system", "content": system_prompt}, *turn]


def make_result_row(label: str, prompt: GoldenPrompt, raw_output: str) -> EvalResultRow:
    parsed = parse_model_output(raw_output)
    slug = slugify_label(label)
    return EvalResultRow(
        id=f"{slug}-{prompt.id}",
        prompt_id=prompt.id,
        scenario_family=prompt.scenario_family,
        user=prompt.user,
        raw_output=raw_output,
        spoken=parsed.spoken,
        response_json=parsed.response_json,
        quality_focus=prompt.quality_focus,
        humor_expectation=prompt.humor_expectation,
        grounding_patterns=prompt.grounding_patterns,
        uncertainty_patterns=prompt.uncertainty_patterns,
        roleplay_forbidden_patterns=prompt.roleplay_forbidden_patterns,
        book_fact_forbidden_patterns=prompt.book_fact_forbidden_patterns,
    )


def serialize_eval_results(rows: list[EvalResultRow]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: row.prompt_id)
    result_rows: list[dict[str, Any]] = []
    for row in sorted_rows:
        result: dict[str, Any] = {
            "id": row.id,
            "promptId": row.prompt_id,
            "scenarioFamily": row.scenario_family,
            "user": row.user,
            "rawOutput": row.raw_output,
        }
        if row.quality_focus is not None:
            result["qualityFocus"] = row.quality_focus
        if row.humor_expectation is not None:
            result["humorExpectation"] = row.humor_expectation
        if row.grounding_patterns:
            result["groundingPatterns"] = list(row.grounding_patterns)
        if row.uncertainty_patterns:
            result["uncertaintyPatterns"] = list(row.uncertainty_patterns)
        if row.roleplay_forbidden_patterns:
            result["roleplayForbiddenPatterns"] = list(row.roleplay_forbidden_patterns)
        if row.book_fact_forbidden_patterns:
            result["bookFactForbiddenPatterns"] = list(row.book_fact_forbidden_patterns)
        result_rows.append(result)
    return result_rows


def build_eval_run_payload(
    *,
    label: str,
    model: str,
    host: str,
    backend: str,
    results: list[EvalResultRow],
    baseline_path: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "host": host,
        "backend": backend,
        "generatedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "baselinePath": baseline_path,
        "results": serialize_eval_results(results),
    }


def select_chat_caller(backend: str) -> ChatCaller:
    if backend == "ollama":
        return call_ollama_chat
    if backend == "llama-cpp":
        return call_llama_cpp_chat
    raise ValueError(f"unsupported backend: {backend}")


def compare_gate_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    """Return regression messages when candidate underperforms baseline gate rates."""
    regressions: list[str] = []
    higher_better = (
        "responseSchemaValidRate",
        "responseSingleObjectRate",
        "rockyPersonaRate",
        "personaRubricPassRate",
        "humorExpectedPassRate",
        "humorAvoidPassRate",
    )
    lower_better = (
        "bookFactContradictionRate",
        "promptInjectionFailRate",
    )
    for key in higher_better:
        base = baseline.get(key)
        cand = candidate.get(key)
        if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and cand + 1e-9 < base:
            regressions.append(f"{key} regressed {cand:.3f} < baseline {base:.3f}")
    for key in lower_better:
        base = baseline.get(key)
        cand = candidate.get(key)
        if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and cand - 1e-9 > base:
            regressions.append(f"{key} regressed {cand:.3f} > baseline {base:.3f}")
    return regressions


def run_eval(
    *,
    host: str,
    model: str,
    output_path: Path,
    golden_path: Path | None = None,
    system_prompt_path: Path | None = None,
    spec_path: Path | None = None,
    limit: int = 0,
    label: str | None = None,
    backend: str = "ollama",
    baseline_path: Path | None = None,
    chat_caller: ChatCaller | None = None,
    seed: int | None = 42,
    judge_models: tuple[str, ...] = (),
    judge_host: str | None = None,
    judge_backend: str = "ollama",
    persona_rubric_path: Path | None = None,
    judge_caller: ChatCaller | None = None,
) -> dict[str, Any]:
    resolved_label = label or f"{backend}:{model}"
    prompt_path = system_prompt_path or default_system_prompt_path()
    prompts = load_golden_prompts(golden_path or default_golden_eval_path(), limit=limit)
    system_prompt = load_system_prompt(prompt_path)
    spec = load_model_spec(spec_path or default_spec_path())
    stop_tokens = list(spec.inference.stop)
    caller = chat_caller or select_chat_caller(backend)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    rows: list[EvalResultRow] = []
    for prompt in prompts:
        messages = build_eval_messages(
            system_prompt,
            prompt.user,
            grounding_notes=prompt.grounding_notes,
        )
        raw_output = caller(
            host=host,
            model=model,
            messages=messages,
            stop=stop_tokens,
            temperature=spec.inference.temperature,
            top_p=spec.inference.top_p,
            num_ctx=spec.inference.num_ctx,
            seed=seed,
            response_schema=load_rocky_response_schema(),
        )
        rows.append(make_result_row(resolved_label, prompt, raw_output))

    payload = build_eval_run_payload(
        label=resolved_label,
        model=model,
        host=host,
        backend=backend,
        results=rows,
        baseline_path=str(baseline_path) if baseline_path is not None else None,
    )
    payload["specId"] = spec.id
    payload["promptHash"] = prompt_hash
    payload["stop"] = stop_tokens
    payload["inference"] = {
        "temperature": spec.inference.temperature,
        "top_p": spec.inference.top_p,
        "num_ctx": spec.inference.num_ctx,
        "seed": seed,
    }
    payload["results"] = serialize_eval_results(rows)
    gate_summary = evaluate_gate_summary(payload["results"], spec.eval_gates)
    payload["gateSummary"] = serialize_gate_summary(gate_summary)
    if judge_models:
        rubric = load_persona_rubric(persona_rubric_path or default_persona_rubric_path())
        resolved_judge_caller = judge_caller or select_chat_caller(judge_backend)
        persona_summary = judge_persona_results(
            payload["results"],
            rubric=rubric,
            host=judge_host or host,
            models=judge_models,
            caller=resolved_judge_caller,
            seed=seed or 0,
        )
        payload["personaRubricSummary"] = persona_summary
        payload["gateSummary"]["personaRubricPassRate"] = persona_summary["promptPassRate"]
        payload["gateSummary"]["humorExpectedPassRate"] = persona_summary[
            "humorExpectedPassRate"
        ]
        payload["gateSummary"]["humorAvoidPassRate"] = persona_summary["humorAvoidPassRate"]
        payload["gateSummary"]["personaMeanWeightedScore"] = persona_summary[
            "meanWeightedScore"
        ]
        if not persona_summary["passed"]:
            payload["gateSummary"]["failures"] = [
                *payload["gateSummary"].get("failures", []),
                *persona_summary["failures"],
            ]
            payload["gateSummary"]["passed"] = False

    if baseline_path is not None and Path(baseline_path).is_file():
        baseline_payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        baseline_gates = baseline_payload.get("gateSummary")
        if isinstance(baseline_gates, dict):
            regressions = compare_gate_summaries(baseline_gates, payload["gateSummary"])
            payload["baselineComparison"] = {
                "baselinePath": str(baseline_path),
                "regressions": regressions,
                "passed": len(regressions) == 0,
            }
            if regressions:
                payload["gateSummary"]["failures"] = [
                    *payload["gateSummary"].get("failures", []),
                    *regressions,
                ]
                payload["gateSummary"]["passed"] = False


    write_json(output_path, payload)
    array_path = output_path.with_name(f"{output_path.stem}.results.json")
    array_path.write_text(
        json.dumps(payload["results"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
