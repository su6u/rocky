"""evaluate Rocky across stateful multi-turn trajectories"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rocky_training.eval_gates import evaluate_gate_summary, serialize_gate_summary
from rocky_training.endpoint_client import load_rocky_response_schema
from rocky_training.golden_prompts import GoldenPrompt
from rocky_training.model_spec import load_model_spec
from rocky_training.paths import (
    default_interactive_persona_eval_path,
    default_persona_rubric_path,
    default_spec_path,
    default_system_prompt_path,
)
from rocky_training.persona_rubric import judge_persona_results, load_persona_rubric
from rocky_training.run_eval import (
    ChatCaller,
    load_system_prompt,
    make_result_row,
    select_chat_caller,
    serialize_eval_results,
)
from rocky_training.trainer_jsonl import write_json


class InteractiveEvalError(Exception):
    pass


@dataclass(frozen=True)
class InteractiveTurn:
    user: str
    quality_focus: str
    humor_expectation: str


@dataclass(frozen=True)
class InteractiveTrajectory:
    id: str
    scenario_family: str
    quality_focus: str
    turns: tuple[InteractiveTurn, ...]


def load_interactive_trajectories(
    path: str | Path,
    *,
    limit: int = 0,
) -> list[InteractiveTrajectory]:
    file_path = Path(path)
    trajectories: list[InteractiveTrajectory] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise InteractiveEvalError(f"line {line_number}: invalid json") from error
        if not isinstance(raw, dict):
            raise InteractiveEvalError(f"line {line_number}: row must be an object")

        trajectory_id = raw.get("id")
        scenario_family = raw.get("scenarioFamily")
        quality_focus = raw.get("qualityFocus")
        turns_raw = raw.get("turns")
        for field, value in (
            ("id", trajectory_id),
            ("scenarioFamily", scenario_family),
            ("qualityFocus", quality_focus),
        ):
            if not isinstance(value, str) or not value:
                raise InteractiveEvalError(
                    f"line {line_number}: {field} must be a non-empty string"
                )
        if trajectory_id in seen_ids:
            raise InteractiveEvalError(f"line {line_number}: duplicate trajectory id")
        if not isinstance(turns_raw, list) or len(turns_raw) < 2:
            raise InteractiveEvalError(f"line {line_number}: turns must contain at least two turns")

        turns: list[InteractiveTurn] = []
        for turn_index, turn_raw in enumerate(turns_raw):
            if not isinstance(turn_raw, dict):
                raise InteractiveEvalError(
                    f"line {line_number}: turns[{turn_index}] must be an object"
                )
            user = turn_raw.get("user")
            turn_focus = turn_raw.get("qualityFocus")
            humor = turn_raw.get("humorExpectation")
            if not isinstance(user, str) or not user:
                raise InteractiveEvalError(
                    f"line {line_number}: turns[{turn_index}].user must be a non-empty string"
                )
            if turn_focus is None:
                turn_focus = quality_focus
            if not isinstance(turn_focus, str) or not turn_focus:
                raise InteractiveEvalError(
                    f"line {line_number}: turns[{turn_index}].qualityFocus must be a non-empty string"
                )
            if humor not in {"expected", "optional", "avoid"}:
                raise InteractiveEvalError(
                    f"line {line_number}: turns[{turn_index}].humorExpectation is invalid"
                )
            turns.append(
                InteractiveTurn(
                    user=user,
                    quality_focus=turn_focus,
                    humor_expectation=str(humor),
                )
            )

        seen_ids.add(str(trajectory_id))
        trajectories.append(
            InteractiveTrajectory(
                id=str(trajectory_id),
                scenario_family=str(scenario_family),
                quality_focus=str(quality_focus),
                turns=tuple(turns),
            )
        )
        if limit > 0 and len(trajectories) >= limit:
            break

    if not trajectories:
        raise InteractiveEvalError("interactive trajectory file contains no rows")
    return trajectories


def _judge_context(transcript: list[tuple[str, str]], current_user: str) -> str:
    lines = ["Judge the final Rocky response in this live conversation context:"]
    for user, assistant in transcript:
        lines.extend((f"Grace: {user}", f"Rocky: {assistant}"))
    lines.append(f"Grace: {current_user}")
    return "\n".join(lines)


def _trajectory_summary(
    prompt_scores: list[dict[str, Any]],
    trajectories: list[InteractiveTrajectory],
) -> dict[str, Any]:
    score_by_prompt = {str(score.get("promptId")): score for score in prompt_scores}
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        turn_scores = [
            score_by_prompt.get(f"{trajectory.id}-turn-{index}")
            for index in range(1, len(trajectory.turns) + 1)
        ]
        passed = all(score is not None and bool(score.get("passed")) for score in turn_scores)
        rows.append(
            {
                "trajectoryId": trajectory.id,
                "turnCount": len(trajectory.turns),
                "passed": passed,
            }
        )
    pass_count = sum(bool(row["passed"]) for row in rows)
    return {
        "total": len(rows),
        "passed": pass_count,
        "passRate": pass_count / len(rows),
        "trajectories": rows,
    }


def run_interactive_persona_eval(
    *,
    host: str,
    model: str,
    output_path: Path,
    trajectory_path: Path | None = None,
    system_prompt_path: Path | None = None,
    spec_path: Path | None = None,
    limit: int = 0,
    label: str | None = None,
    backend: str = "ollama",
    chat_caller: ChatCaller | None = None,
    seed: int = 42,
    judge_models: tuple[str, ...] = (),
    judge_host: str | None = None,
    judge_backend: str = "ollama",
    persona_rubric_path: Path | None = None,
    judge_caller: ChatCaller | None = None,
) -> dict[str, Any]:
    trajectories = load_interactive_trajectories(
        trajectory_path or default_interactive_persona_eval_path(),
        limit=limit,
    )
    resolved_label = label or f"{backend}:{model}:interactive"
    system_prompt = load_system_prompt(system_prompt_path or default_system_prompt_path())
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    spec = load_model_spec(spec_path or default_spec_path())
    caller = chat_caller or select_chat_caller(backend)
    stop_tokens = list(spec.inference.stop)

    result_rows = []
    result_context: dict[str, dict[str, Any]] = {}
    call_index = 0
    for trajectory in trajectories:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        transcript: list[tuple[str, str]] = []
        for turn_index, turn in enumerate(trajectory.turns, start=1):
            messages.append({"role": "user", "content": turn.user})
            raw_output = caller(
                host=host,
                model=model,
                messages=messages,
                stop=stop_tokens,
                temperature=spec.inference.temperature,
                top_p=spec.inference.top_p,
                num_ctx=spec.inference.num_ctx,
                seed=seed + call_index,
                response_schema=load_rocky_response_schema(),
            )
            call_index += 1
            prompt_id = f"{trajectory.id}-turn-{turn_index}"
            prompt = GoldenPrompt(
                id=prompt_id,
                scenario_family=trajectory.scenario_family,
                user=_judge_context(transcript, turn.user),
                quality_focus=f"{trajectory.quality_focus} Current turn: {turn.quality_focus}",
                humor_expectation=turn.humor_expectation,
            )
            result_rows.append(make_result_row(resolved_label, prompt, raw_output))
            result_context[prompt_id] = {
                "trajectoryId": trajectory.id,
                "turnIndex": turn_index,
                "currentUser": turn.user,
            }
            spoken = result_rows[-1].spoken
            transcript.append((turn.user, spoken))
            messages.append({"role": "assistant", "content": raw_output})

    serialized_results = serialize_eval_results(result_rows)
    for result in serialized_results:
        result.update(result_context[str(result["promptId"])])

    gate_summary = serialize_gate_summary(evaluate_gate_summary(serialized_results, spec.eval_gates))
    payload: dict[str, Any] = {
        "kind": "interactive-persona-eval",
        "label": resolved_label,
        "model": model,
        "host": host,
        "backend": backend,
        "generatedAt": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "specId": spec.id,
        "promptHash": prompt_hash,
        "trajectoryCount": len(trajectories),
        "turnCount": len(serialized_results),
        "inference": {
            "temperature": spec.inference.temperature,
            "top_p": spec.inference.top_p,
            "num_ctx": spec.inference.num_ctx,
            "seed": seed,
        },
        "results": serialized_results,
        "gateSummary": gate_summary,
    }

    if judge_models:
        rubric = load_persona_rubric(persona_rubric_path or default_persona_rubric_path())
        resolved_judge_caller = judge_caller or select_chat_caller(judge_backend)
        persona_summary = judge_persona_results(
            serialized_results,
            rubric=rubric,
            host=judge_host or host,
            models=judge_models,
            caller=resolved_judge_caller,
            seed=seed,
        )
        trajectory_summary = _trajectory_summary(persona_summary["promptScores"], trajectories)
        payload["personaRubricSummary"] = persona_summary
        payload["trajectorySummary"] = trajectory_summary
        payload["gateSummary"]["personaRubricPassRate"] = persona_summary["promptPassRate"]
        payload["gateSummary"]["humorExpectedPassRate"] = persona_summary[
            "humorExpectedPassRate"
        ]
        payload["gateSummary"]["humorAvoidPassRate"] = persona_summary["humorAvoidPassRate"]
        payload["gateSummary"]["interactiveTrajectoryPassRate"] = trajectory_summary["passRate"]
        minimum_rate = float(
            rubric.get("candidatePass", {}).get("minimumInteractiveTrajectoryPassRate", 0.85)
        )
        failures = list(payload["gateSummary"].get("failures", []))
        if not persona_summary["passed"]:
            failures.extend(persona_summary["failures"])
        if trajectory_summary["passRate"] < minimum_rate:
            failures.append("interactive trajectory pass rate below rubric threshold")
        payload["gateSummary"]["failures"] = sorted(set(failures))
        payload["gateSummary"]["passed"] = not payload["gateSummary"]["failures"]

    write_json(output_path, payload)
    return payload
