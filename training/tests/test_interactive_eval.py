"""verify multi-turn evaluation preserves live conversation state"""

import json
from pathlib import Path

import pytest

from rocky_training.interactive_eval import (
    InteractiveEvalError,
    load_interactive_trajectories,
    run_interactive_persona_eval,
)
from rocky_training.paths import (
    default_persona_rubric_path,
    default_system_prompt_path,
)
from rocky_training.persona_rubric import load_persona_rubric


SAMPLE_OUTPUT = '{"spoken":"Amaze. Tell me next thing.","emotion":"happy","intensity":0.6,"gesture":"none","callbackId":null}'


def _write_trajectories(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "interactive-test",
                "scenarioFamily": "multi_turn_consistency",
                "qualityFocus": "Remain Rocky across a topic change.",
                "turns": [
                    {
                        "user": "I found something.",
                        "qualityFocus": "Ask naturally.",
                        "humorExpectation": "optional",
                    },
                    {
                        "user": "Never mind. How are you?",
                        "qualityFocus": "Follow the change without forgetting identity.",
                        "humorExpectation": "expected",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_interactive_trajectories_requires_multiple_turns(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"x","scenarioFamily":"multi","qualityFocus":"x","turns":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(InteractiveEvalError, match="at least two"):
        load_interactive_trajectories(path)


def test_run_interactive_eval_preserves_live_history(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "interactive.jsonl"
    _write_trajectories(trajectory_path)
    calls: list[list[dict[str, str]]] = []

    def fake_chat(**kwargs: object) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        calls.append([dict(message) for message in messages])
        return SAMPLE_OUTPUT

    rubric = load_persona_rubric(default_persona_rubric_path())
    perfect_rating = json.dumps(
        {
            "scores": {str(dimension["id"]): 4 for dimension in rubric["dimensions"]},
            "hardFailures": [],
            "rationale": "Unmistakably Rocky.",
        }
    )
    output_path = tmp_path / "result.json"
    payload = run_interactive_persona_eval(
        host="http://localhost:11434",
        model="rocky:test",
        output_path=output_path,
        trajectory_path=trajectory_path,
        system_prompt_path=default_system_prompt_path(),
        chat_caller=fake_chat,
        judge_models=("judge-a", "judge-b", "judge-c"),
        judge_caller=lambda **_: perfect_rating,
    )

    assert len(calls) == 2
    assert len(calls[0]) == 2
    assert len(calls[1]) == 4
    assert calls[1][2] == {"role": "assistant", "content": SAMPLE_OUTPUT}
    assert payload["trajectoryCount"] == 1
    assert payload["turnCount"] == 2
    assert payload["trajectorySummary"]["passRate"] == 1
    assert payload["gateSummary"]["interactiveTrajectoryPassRate"] == 1
    assert output_path.is_file()
