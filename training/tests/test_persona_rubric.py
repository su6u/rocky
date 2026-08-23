"""validate the v1 independent persona evaluation rubric"""

import json

import pytest

from rocky_training.paths import default_persona_rubric_path
from rocky_training.persona_rubric import (
    PersonaRubricError,
    build_persona_judge_messages,
    judge_persona_results,
    load_persona_rubric,
    parse_persona_rating,
)


def _perfect_rating(rubric: dict[str, object]) -> str:
    dimensions = rubric["dimensions"]
    assert isinstance(dimensions, list)
    scores = {str(dimension["id"]): 4 for dimension in dimensions}
    return json.dumps({"scores": scores, "hardFailures": [], "rationale": "Strong Rocky response."})


def _result() -> dict[str, object]:
    return {
        "promptId": "persona-test",
        "scenarioFamily": "casual_humor",
        "user": "Why do humans name ships?",
        "rawOutput": (
            '{"spoken":"Ship moves and saves humans. Chair only waits. Still unfair to chair.",'
            '"emotion":"happy","intensity":0.6,"gesture":"tap_carapace","callbackId":null}'
        ),
        "qualityFocus": "literal cultural humor",
        "humorExpectation": "expected",
    }


def test_build_persona_judge_messages_includes_prompt_policy() -> None:
    rubric = load_persona_rubric(default_persona_rubric_path())
    messages = build_persona_judge_messages(rubric, _result())

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "humorExpectation" in messages[1]["content"]
    assert "Why do humans name ships?" in messages[1]["content"]


def test_persona_rubric_rejects_ai_disclaimers_but_allows_playful_companionship() -> None:
    rubric = load_persona_rubric(default_persona_rubric_path())
    assert rubric["version"] == "rocky-persona-v1"
    voice = next(
        dimension
        for dimension in rubric["dimensions"]
        if dimension["id"] == "voice_fidelity"
    )
    action_failure = rubric["hardFailureDefinitions"][
        "invented_memory_sensory_access_or_external_action"
    ]

    assert "software" in voice["criteria"]
    assert "consequential real-world action" in action_failure


def test_signature_beats_are_trigger_bound_and_batch_limited() -> None:
    rubric = load_persona_rubric(default_persona_rubric_path())
    policy = rubric["signatureBeatPolicy"]

    assert "genuinely confused" in policy["questionSuffix"]["allowedWhen"]
    assert "real discovery" in policy["amazeRepetition"]["allowedWhen"]
    assert "Immediate danger" in policy["badRepetition"]["allowedWhen"]
    assert policy["batchTargetsPer500Rows"] == {
        "reviewedMovieCallbacks": 15,
        "questionSuffixMinimum": 50,
        "questionSuffixMaximum": 140,
        "amazeRepetitionMinimum": 5,
        "amazeRepetitionMaximum": 15,
        "badRepetitionMinimum": 5,
        "badRepetitionMaximum": 15,
    }


def test_parse_persona_rating_requires_every_dimension() -> None:
    rubric = load_persona_rubric(default_persona_rubric_path())

    with pytest.raises(PersonaRubricError, match="match rubric dimensions"):
        parse_persona_rating(
            '{"scores":{"voice_fidelity":4},"hardFailures":[],"rationale":"short"}',
            rubric=rubric,
            rater="judge-a",
        )


def test_judge_persona_results_aggregates_three_distinct_raters() -> None:
    rubric = load_persona_rubric(default_persona_rubric_path())
    calls: list[str] = []

    def fake_judge(**kwargs: object) -> str:
        calls.append(str(kwargs["model"]))
        return _perfect_rating(rubric)

    summary = judge_persona_results(
        [_result()],
        rubric=rubric,
        host="http://localhost:11434",
        models=("judge-a", "judge-b", "judge-c"),
        caller=fake_judge,
    )

    assert calls == ["judge-a", "judge-b", "judge-c"]
    assert summary["promptPassRate"] == 1
    assert summary["humorExpectedPassRate"] == 1
    assert summary["passed"] is True


def test_judge_persona_results_rejects_non_independent_raters() -> None:
    rubric = load_persona_rubric(default_persona_rubric_path())

    with pytest.raises(PersonaRubricError, match="distinct judge models"):
        judge_persona_results(
            [_result()],
            rubric=rubric,
            host="http://localhost:11434",
            models=("judge-a", "judge-a", "judge-a"),
            caller=lambda **_: _perfect_rating(rubric),
        )
