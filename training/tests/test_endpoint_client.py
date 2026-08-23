"""verify inference adapters request constrained Rocky responses"""

from unittest.mock import patch

from rocky_training.endpoint_client import call_ollama_chat, load_rocky_response_schema


def test_call_ollama_chat_passes_schema_and_stop_tokens() -> None:
    with patch("rocky_training.endpoint_client._post_json") as post_json:
        post_json.return_value = {"message": {"content": "ok"}}
        call_ollama_chat(
            host="http://localhost:11434",
            model="rocky:v1",
            messages=[{"role": "user", "content": "hi"}],
            stop=["<turn|>"],
            response_schema=load_rocky_response_schema(),
        )

    payload = post_json.call_args.args[1]
    assert payload["options"]["stop"] == ["<turn|>"]
    assert payload["format"]["required"] == [
        "spoken",
        "emotion",
        "intensity",
        "gesture",
        "callbackId",
    ]
    assert payload["think"] is False
