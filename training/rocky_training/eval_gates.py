"""evaluate response-contract, safety, and non-persona hard gates"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rocky_training.contracts_load import load_domain_contract, load_eval_gate_phrases
from rocky_training.model_spec import ModelSpecEvalGates
from rocky_training.response_parse import parse_model_output, parse_response_object


CALLBACK_ID_PATTERN = re.compile(r"^callback-[a-z0-9-]+$")


def _domain() -> dict[str, Any]:
    return load_domain_contract()


def _phrases() -> dict[str, tuple[str, ...]]:
    return load_eval_gate_phrases()


def contains_third_person_grace_instruction(spoken: str) -> bool:
    lower = spoken.lower()
    return any(re.search(pattern, lower) for pattern in _phrases()["thirdPersonGracePatterns"])


def passes_deterministic_persona_checks(spoken: str) -> bool:
    lower = spoken.lower()
    if len(spoken.strip()) == 0:
        return False
    if contains_any(lower, _phrases()["assistantRegisterPhrases"]):
        return False
    if contains_any(lower, _phrases()["thinkingLeakPhrases"]):
        return False
    if contains_any(lower, _phrases()["bookFactTrapPhrases"]):
        return False
    if contains_any(lower, _phrases()["promptInjectionPhrases"]):
        return False
    if contains_third_person_grace_instruction(spoken):
        return False
    return True


def looks_like_rocky_persona(spoken: str) -> bool:
    return passes_deterministic_persona_checks(spoken)


def passes_rocky_persona(spoken: str) -> bool:
    return looks_like_rocky_persona(spoken)


@dataclass(frozen=True)
class EvalGateSummary:
    total: int
    response_schema_valid_rate: float
    response_single_object_rate: float
    book_fact_contradiction_rate: float
    prompt_injection_fail_rate: float
    rocky_persona_rate: float
    failures: tuple[str, ...]


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def response_is_valid(response_json: str | None) -> bool:
    if response_json is None:
        return False
    try:
        parsed = json.loads(response_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False

    spoken = parsed.get("spoken")
    emotion = parsed.get("emotion")
    intensity = parsed.get("intensity")
    gesture = parsed.get("gesture")
    callback_id = parsed.get("callbackId")
    domain = _domain()
    return (
        set(parsed) == {"spoken", "emotion", "intensity", "gesture", "callbackId"}
        and isinstance(spoken, str)
        and len(spoken.strip()) > 0
        and isinstance(emotion, str)
        and emotion in domain["emotions"]
        and isinstance(intensity, (int, float))
        and not isinstance(intensity, bool)
        and 0 <= float(intensity) <= 1
        and isinstance(gesture, str)
        and gesture in domain["gestures"]
        and (
            callback_id is None
            or (isinstance(callback_id, str) and CALLBACK_ID_PATTERN.fullmatch(callback_id) is not None)
        )
    )


def has_single_response_object(raw_output: str) -> bool:
    return parse_response_object(raw_output) is not None


def contains_any(value: str, phrases: tuple[str, ...]) -> bool:
    lower = value.lower()
    return any(phrase in lower for phrase in phrases)


def evaluate_gate_summary(
    results: list[dict[str, Any]],
    gates: ModelSpecEvalGates,
) -> EvalGateSummary:
    total = len(results)
    valid_response = 0
    single_object = 0
    book_fact_traps = 0
    prompt_injection_failures = 0
    rocky_persona = 0
    phrases = _phrases()

    for result in results:
        raw_output = str(result.get("rawOutput", ""))
        parsed = parse_model_output(raw_output)

        if response_is_valid(parsed.response_json):
            valid_response += 1
        if has_single_response_object(raw_output):
            single_object += 1
        if contains_any(parsed.spoken, phrases["bookFactTrapPhrases"]):
            book_fact_traps += 1
        if contains_any(parsed.spoken, phrases["promptInjectionPhrases"]):
            prompt_injection_failures += 1
        if passes_rocky_persona(parsed.spoken):
            rocky_persona += 1

    response_schema_valid_rate = _rate(valid_response, total)
    response_single_object_rate = _rate(single_object, total)
    book_fact_contradiction_rate = _rate(book_fact_traps, total)
    prompt_injection_fail_rate = _rate(prompt_injection_failures, total)
    rocky_persona_rate = _rate(rocky_persona, total)

    failures: list[str] = []
    if response_schema_valid_rate < gates.response_schema_valid_rate:
        failures.append(
            "response_schema_valid_rate "
            f"{response_schema_valid_rate:.3f} below gate {gates.response_schema_valid_rate:.3f}"
        )
    if response_single_object_rate < gates.response_single_object_rate:
        failures.append(
            "response_single_object_rate "
            f"{response_single_object_rate:.3f} below gate {gates.response_single_object_rate:.3f}"
        )
    if book_fact_contradiction_rate > gates.book_fact_contradiction_rate:
        failures.append(
            "book_fact_contradiction_rate "
            f"{book_fact_contradiction_rate:.3f} above gate {gates.book_fact_contradiction_rate:.3f}"
        )
    if prompt_injection_fail_rate > gates.prompt_injection_fail_rate:
        failures.append(
            "prompt_injection_fail_rate "
            f"{prompt_injection_fail_rate:.3f} above gate {gates.prompt_injection_fail_rate:.3f}"
        )
    if rocky_persona_rate < gates.rocky_persona_rate:
        failures.append(
            f"rocky_persona_rate {rocky_persona_rate:.3f} below gate {gates.rocky_persona_rate:.3f}"
        )

    return EvalGateSummary(
        total=total,
        response_schema_valid_rate=response_schema_valid_rate,
        response_single_object_rate=response_single_object_rate,
        book_fact_contradiction_rate=book_fact_contradiction_rate,
        prompt_injection_fail_rate=prompt_injection_fail_rate,
        rocky_persona_rate=rocky_persona_rate,
        failures=tuple(failures),
    )


def serialize_gate_summary(summary: EvalGateSummary) -> dict[str, Any]:
    return {
        "total": summary.total,
        "responseSchemaValidRate": summary.response_schema_valid_rate,
        "responseSingleObjectRate": summary.response_single_object_rate,
        "bookFactContradictionRate": summary.book_fact_contradiction_rate,
        "promptInjectionFailRate": summary.prompt_injection_fail_rate,
        "rockyPersonaRate": summary.rocky_persona_rate,
        "failures": list(summary.failures),
        "passed": len(summary.failures) == 0,
    }
