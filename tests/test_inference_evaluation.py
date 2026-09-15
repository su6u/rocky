import io
import json

import pytest

from rocky.data import DATA, read_jsonl
from rocky.evaluation import evaluate
from rocky.inference import GenerationSettings, InferenceClient
from rocky.session import SceneSession


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(row["content"] for row in messages)

    def encode(self, text, **kwargs):
        return text.split()


class Client:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def complete(self, messages):
        self.calls.append(list(messages))
        if self.fail_first and len(self.calls) == 1:
            raise ValueError("completion did not finish normally: length")
        return "Recorded reply."


def test_budget_stops_transport_and_preserves_history(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("HTTP must not run after an oversized prompt")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    session = SceneSession()
    client = InferenceClient("http://localhost:8000/v1", "rocky", Tokenizer(),
                             GenerationSettings(context_tokens=20, max_tokens=10))
    with pytest.raises(ValueError, match="history was not dropped"):
        client.complete(session.prompt("Hello."))
    assert session.history == []


@pytest.mark.parametrize("finish", ["length", "tool_calls", None])
def test_abnormal_completion_is_not_a_valid_reply(monkeypatch, finish):
    def response(request, **kwargs):
        body = json.loads(request.data)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["temperature"] == 0
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "Partial"}, "finish_reason": finish}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen", response)
    client = InferenceClient("http://localhost:8000/v1", "rocky", Tokenizer())
    with pytest.raises(ValueError, match="did not finish normally"):
        client.complete([{"role": "system", "content": "Scene"}, {"role": "user", "content": "Hello"}])


def test_trajectory_records_generated_history_and_continues_after_case_error(tmp_path):
    path = tmp_path / "trials.jsonl"
    client = Client(fail_first=True)
    result = evaluate("http://localhost/v1", "rocky", path, suite="trajectories", client=client)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert result["cases"] == 6 and result["errors"] == 1
    assert result["status"] == "completed_with_failures" and result["persona_score"] is None
    assert len(client.calls) > 6
    assert records[0]["generation"]["temperature"] == 0
    assert records[1]["type"] == "error"
    replies = [row for row in records if row["type"] == "reply"]
    assert replies[1]["messages"][-2] == {"role": "assistant", "content": "Recorded reply."}
    assert "must" in replies[1] and "latency_seconds" in replies[1]
    with pytest.raises(FileExistsError):
        evaluate("http://localhost/v1", "rocky", path, suite="trajectories", client=client)


def test_probe_runtime_boundary_is_not_sent_to_model(tmp_path):
    client = Client()
    result = evaluate("http://localhost/v1", "rocky", tmp_path / "probes.jsonl", suite="probes", client=client)
    assert result["runtime_checks"] == 1 and result["errors"] == 0
    assert result["replies"] == len(read_jsonl(DATA / "eval/probes.jsonl")) - 1
    assert len(client.calls) == result["replies"]
    assert not any("whole life story" in row["content"] for call in client.calls for row in call)


def test_malformed_speaker_output_cannot_enter_session_history():
    session = SceneSession()
    with pytest.raises(ValueError, match="history unchanged"):
        session.accept("Hello.", "Grace: Hello.")
    assert session.history == []
