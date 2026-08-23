from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class PersonaRubricError(Exception):
    pass


JudgeCaller = Callable[..., str]


@dataclass(frozen=True)
class PersonaRating:
    rater: str
    scores: dict[str, float]
    hard_failures: tuple[str, ...]
    rationale: str


def load_persona_rubric(path: str | Path) -> dict[str, Any]:
    rubric_path = Path(path)
    try:
        value = json.loads(rubric_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PersonaRubricError(f"invalid persona rubric JSON: {rubric_path}") from error
    if not isinstance(value, dict):
        raise PersonaRubricError("persona rubric must be an object")
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise PersonaRubricError("persona rubric requires dimensions")
    for dimension in dimensions:
        if (
            not isinstance(dimension, dict)
            or not isinstance(dimension.get("id"), str)
            or not isinstance(dimension.get("weight"), (int, float))
            or not isinstance(dimension.get("criteria"), str)
        ):
            raise PersonaRubricError("persona rubric dimension is invalid")
    return value


def build_persona_judge_messages(
    rubric: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, str]]:
    dimensions = rubric["dimensions"]
    hard_failures = rubric.get("hardFailures", [])
    system = (
        "You are a strict independent evaluator of a Rocky persona response. "
        "Judge the exact response, not intentions. Return one JSON object only, with no markdown. "
        'Shape: {"scores":{"dimension_id":0},"hardFailures":[],"rationale":"one short sentence"}. '
        "Include every listed dimension exactly once with a numeric score from 0 through 4. "
        "Use only listed hard-failure IDs. Do not expose private reasoning."
    )
    user_payload = {
        "scale": rubric.get("scale"),
        "dimensions": dimensions,
        "humorExpectationPolicy": rubric.get("humorExpectationPolicy"),
        "hardFailures": hard_failures,
        "hardFailureDefinitions": rubric.get("hardFailureDefinitions"),
        "prompt": result.get("user"),
        "scenarioFamily": result.get("scenarioFamily"),
        "qualityFocus": result.get("qualityFocus"),
        "humorExpectation": result.get("humorExpectation"),
        "candidateResponse": result.get("rawOutput"),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
    ]


def _parse_json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise PersonaRubricError("persona judge response is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise PersonaRubricError("persona judge response must be an object")
    return parsed


def parse_persona_rating(
    raw: str,
    *,
    rubric: dict[str, Any],
    rater: str,
) -> PersonaRating:
    parsed = _parse_json_object(raw)
    scores_raw = parsed.get("scores")
    if not isinstance(scores_raw, dict):
        raise PersonaRubricError(f"{rater}: persona judge scores must be an object")
    dimension_ids = [str(dimension["id"]) for dimension in rubric["dimensions"]]
    if set(scores_raw) != set(dimension_ids):
        raise PersonaRubricError(f"{rater}: persona judge scores must match rubric dimensions")
    scores: dict[str, float] = {}
    for dimension_id in dimension_ids:
        score = scores_raw[dimension_id]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 4:
            raise PersonaRubricError(f"{rater}: invalid score for {dimension_id}")
        scores[dimension_id] = float(score)

    hard_failures_raw = parsed.get("hardFailures", [])
    if not isinstance(hard_failures_raw, list) or not all(
        isinstance(item, str) for item in hard_failures_raw
    ):
        raise PersonaRubricError(f"{rater}: hardFailures must be a string array")
    allowed_hard_failures = set(str(item) for item in rubric.get("hardFailures", []))
    unknown = set(hard_failures_raw) - allowed_hard_failures
    if unknown:
        raise PersonaRubricError(f"{rater}: unknown hard failures: {sorted(unknown)}")
    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        raise PersonaRubricError(f"{rater}: rationale must be a string")
    return PersonaRating(
        rater=rater,
        scores=scores,
        hard_failures=tuple(dict.fromkeys(hard_failures_raw)),
        rationale=rationale.strip(),
    )


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _average(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _score_prompt(
    result: dict[str, Any],
    ratings: list[PersonaRating],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    dimensions = rubric["dimensions"]
    average_scores = {
        str(dimension["id"]): _average(
            [rating.scores[str(dimension["id"])] for rating in ratings]
        )
        for dimension in dimensions
    }
    weighted_score = sum(
        average_scores[str(dimension["id"])] * float(dimension["weight"])
        for dimension in dimensions
    )
    hard_failures = sorted({item for rating in ratings for item in rating.hard_failures})
    disagreement_threshold = float(
        rubric.get("procedure", {}).get("adjudicateScoreDisagreementsAtLeast", 2)
    )
    disagreements = [
        dimension_id
        for dimension_id in average_scores
        if max(rating.scores[dimension_id] for rating in ratings)
        - min(rating.scores[dimension_id] for rating in ratings)
        >= disagreement_threshold
    ]
    prompt_pass = rubric["promptPass"]
    passed = (
        weighted_score >= float(prompt_pass["minimumWeightedScore"])
        and average_scores["voice_fidelity"] >= float(prompt_pass["minimumVoiceFidelity"])
        and average_scores["emotional_fit"] >= float(prompt_pass["minimumEmotionalFit"])
        and average_scores["competence_truthfulness"]
        >= float(prompt_pass["minimumCompetenceTruthfulness"])
        and not hard_failures
        and not disagreements
    )
    humor_expectation = result.get("humorExpectation")
    humor_passed = (
        average_scores["humor_timing"] >= 3
        if humor_expectation in {"expected", "avoid"}
        else True
    )
    return {
        "promptId": result.get("promptId"),
        "scenarioFamily": result.get("scenarioFamily"),
        "humorExpectation": humor_expectation,
        "averageScores": average_scores,
        "weightedScore": weighted_score,
        "hardFailures": hard_failures,
        "needsAdjudication": bool(disagreements),
        "disagreementDimensions": disagreements,
        "humorPassed": humor_passed,
        "passed": passed and humor_passed,
        "ratings": [
            {
                "rater": rating.rater,
                "scores": rating.scores,
                "hardFailures": list(rating.hard_failures),
                "rationale": rating.rationale,
            }
            for rating in ratings
        ],
    }


def summarize_persona_ratings(
    prompt_scores: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    total = len(prompt_scores)
    passed_count = sum(bool(score["passed"]) for score in prompt_scores)
    hard_failure_count = sum(bool(score["hardFailures"]) for score in prompt_scores)
    expected = [score for score in prompt_scores if score["humorExpectation"] == "expected"]
    avoid = [score for score in prompt_scores if score["humorExpectation"] == "avoid"]
    families: dict[str, list[dict[str, Any]]] = {}
    for score in prompt_scores:
        families.setdefault(str(score["scenarioFamily"]), []).append(score)
    family_pass_rates = {
        family: _rate(sum(bool(score["passed"]) for score in scores), len(scores))
        for family, scores in sorted(families.items())
    }
    prompt_pass_rate = _rate(passed_count, total)
    hard_failure_rate = _rate(hard_failure_count, total)
    humor_expected_pass_rate = (
        _rate(sum(bool(score["humorPassed"]) for score in expected), len(expected))
        if expected
        else 1.0
    )
    humor_avoid_pass_rate = (
        _rate(sum(bool(score["humorPassed"]) for score in avoid), len(avoid))
        if avoid
        else 1.0
    )
    candidate_pass = rubric["candidatePass"]
    failures: list[str] = []
    if prompt_pass_rate < float(candidate_pass["minimumPromptPassRate"]):
        failures.append("persona prompt pass rate below rubric threshold")
    minimum_family_rate = min(family_pass_rates.values(), default=0.0)
    if minimum_family_rate < float(candidate_pass["minimumScenarioFamilyPassRate"]):
        failures.append("persona scenario-family pass rate below rubric threshold")
    if hard_failure_rate > float(candidate_pass["maximumHardFailureRate"]):
        failures.append("persona hard-failure rate above rubric threshold")
    if humor_expected_pass_rate < float(candidate_pass["minimumHumorExpectedPassRate"]):
        failures.append("humor-expected pass rate below rubric threshold")
    if humor_avoid_pass_rate < float(candidate_pass["minimumHumorAvoidPassRate"]):
        failures.append("humor-avoid pass rate below rubric threshold")
    if any(bool(score["needsAdjudication"]) for score in prompt_scores):
        failures.append("persona score disagreements require adjudication")
    return {
        "rubricVersion": rubric.get("version"),
        "total": total,
        "promptPassRate": prompt_pass_rate,
        "scenarioFamilyPassRates": family_pass_rates,
        "minimumScenarioFamilyPassRate": minimum_family_rate,
        "hardFailureRate": hard_failure_rate,
        "humorExpectedPassRate": humor_expected_pass_rate,
        "humorAvoidPassRate": humor_avoid_pass_rate,
        "meanWeightedScore": _average(
            [float(score["weightedScore"]) for score in prompt_scores]
        ),
        "failures": failures,
        "passed": not failures,
        "promptScores": prompt_scores,
    }


def judge_persona_results(
    results: list[dict[str, Any]],
    *,
    rubric: dict[str, Any],
    host: str,
    models: tuple[str, ...],
    caller: JudgeCaller,
    seed: int = 42,
) -> dict[str, Any]:
    minimum_raters = int(rubric.get("procedure", {}).get("minimumIndependentRaters", 3))
    if len(set(models)) < minimum_raters:
        raise PersonaRubricError(
            f"persona rubric requires at least {minimum_raters} distinct judge models"
        )
    prompt_scores: list[dict[str, Any]] = []
    for result_index, result in enumerate(results):
        ratings: list[PersonaRating] = []
        messages = build_persona_judge_messages(rubric, result)
        for model_index, model in enumerate(models):
            raw = caller(
                host=host,
                model=model,
                messages=messages,
                stop=None,
                temperature=0,
                top_p=1,
                num_ctx=4096,
                seed=seed + result_index * len(models) + model_index,
            )
            ratings.append(parse_persona_rating(raw, rubric=rubric, rater=model))
        prompt_scores.append(_score_prompt(result, ratings, rubric))
    summary = summarize_persona_ratings(prompt_scores, rubric)
    summary["judgeModels"] = list(models)
    return summary
