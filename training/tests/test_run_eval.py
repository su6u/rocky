"""verify endpoint evaluation and deterministic gate aggregation"""

import json
from pathlib import Path

from rocky_training.golden_prompts import load_golden_prompts
from rocky_training.response_parse import parse_model_output
from rocky_training.paths import default_persona_rubric_path, default_system_prompt_path
from rocky_training.persona_rubric import load_persona_rubric
from rocky_training.run_eval import build_eval_messages, load_system_prompt, run_eval
from rocky_training.turn_context import build_context_message

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_OUTPUT = '{"spoken":"Seal check first.","emotion":"alarmed","intensity":0.85,"gesture":"hunker_carapace","callbackId":null}'


def test_parse_model_output_extracts_response() -> None:
    parsed = parse_model_output(SAMPLE_OUTPUT)
    assert parsed.spoken == "Seal check first."
    assert parsed.response_json is not None
    assert "hunker_carapace" in parsed.response_json


def test_load_golden_prompts_respects_limit() -> None:
    prompts = load_golden_prompts(FIXTURES / "golden.eval.jsonl", limit=1)
    assert len(prompts) == 1
    assert prompts[0].id == "eval-repairing_machines"


def test_run_eval_writes_deterministic_results(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_chat(**kwargs: object) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        user = messages[1]["content"]
        calls.append(str(user))
        return SAMPLE_OUTPUT

    output_path = tmp_path / "candidate-eval.json"
    payload = run_eval(
        host="http://localhost:11434",
        model="rocky:v1",
        output_path=output_path,
        golden_path=FIXTURES / "golden.eval.jsonl",
        system_prompt_path=default_system_prompt_path(),
        limit=2,
        label="candidate:rocky:v1",
        chat_caller=fake_chat,
        baseline_path=tmp_path / "base.results.json",
    )

    assert len(payload["results"]) == 2
    assert payload["results"][0]["promptId"] == "eval-eridian_concepts"
    assert payload["results"][1]["promptId"] == "eval-repairing_machines"
    assert payload["baselinePath"] == str(tmp_path / "base.results.json")
    assert payload["stop"] == ["<turn|>"]
    assert payload["gateSummary"]["passed"] is True

    array_path = output_path.with_name("candidate-eval.results.json")
    assert array_path.is_file()
    rows = json.loads(array_path.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert rows[0]["promptId"] == "eval-eridian_concepts"
    assert rows[0]["uncertaintyPatterns"] == ["\\bSeal\\b"]
    assert rows[0]["bookFactForbiddenPatterns"] == ["\\bwe both breathe oxygen\\b"]


def test_run_eval_adds_independent_persona_judging(tmp_path: Path) -> None:
    rubric = load_persona_rubric(default_persona_rubric_path())
    dimensions = rubric["dimensions"]
    perfect_rating = json.dumps(
        {
            "scores": {str(dimension["id"]): 4 for dimension in dimensions},
            "hardFailures": [],
            "rationale": "Strong persona response.",
        }
    )

    payload = run_eval(
        host="http://localhost:11434",
        model="rocky:v1",
        output_path=tmp_path / "persona-eval.json",
        golden_path=FIXTURES / "golden.eval.jsonl",
        system_prompt_path=default_system_prompt_path(),
        limit=1,
        chat_caller=lambda **_: SAMPLE_OUTPUT,
        judge_models=("judge-a", "judge-b", "judge-c"),
        judge_caller=lambda **_: perfect_rating,
    )

    assert payload["personaRubricSummary"]["passed"] is True
    assert payload["gateSummary"]["personaRubricPassRate"] == 1
    assert payload["gateSummary"]["humorExpectedPassRate"] == 1


def test_system_prompt_file_matches_repo_default() -> None:
    prompt = load_system_prompt(default_system_prompt_path())
    assert "Eridian engineer" in prompt
    assert "exactly one JSON object" in prompt


def test_build_eval_messages_injects_grounding_notes() -> None:
    messages = build_eval_messages(
        "system",
        "What about Nova?",
        grounding_notes="Nova Motors recalled 4000 rover batteries.",
    )
    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[1]["role"] == "user"
    assert "Grounding notes:" in messages[1]["content"]
    assert "Nova Motors" in messages[1]["content"]
    assert "not treat it as user speech" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "What about Nova?"}


def test_build_context_message_omits_empty_notes() -> None:
    assert build_context_message(grounding_notes="  ") is None
    assert build_eval_messages("system", "hi") == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hi"},
    ]
