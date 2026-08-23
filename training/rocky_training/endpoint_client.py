"""call serving adapters with optional schema-constrained decoding"""

from __future__ import annotations

import json
from functools import lru_cache
import urllib.error
import urllib.request
from typing import Any

from rocky_training.paths import default_response_schema_path


class EndpointError(Exception):
    pass


@lru_cache(maxsize=1)
def load_rocky_response_schema() -> dict[str, Any]:
    path = default_response_schema_path()
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EndpointError(f"response schema not found: {path}") from error
    except json.JSONDecodeError as error:
        raise EndpointError(f"response schema is invalid JSON: {path}") from error
    if not isinstance(schema, dict):
        raise EndpointError("response schema must be an object")
    return schema


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise EndpointError(f"request failed: {error.code} {detail}") from error
    except urllib.error.URLError as error:
        raise EndpointError(f"request failed: {error.reason}") from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise EndpointError("response was not valid json") from error

    if not isinstance(parsed, dict):
        raise EndpointError("response must be an object")
    return parsed


def call_ollama_chat(
    *,
    host: str,
    model: str,
    messages: list[dict[str, str]],
    stop: list[str] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    response_schema: dict[str, Any] | None = None,
) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    options: dict[str, Any] = {}
    if stop:
        options["stop"] = stop
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if seed is not None:
        options["seed"] = seed
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    if options:
        payload["options"] = options
    if response_schema is not None:
        payload["format"] = response_schema
    parsed = _post_json(url, payload)
    message = parsed.get("message")
    if not isinstance(message, dict):
        raise EndpointError("ollama response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise EndpointError("ollama response missing message.content")
    return content


def call_llama_cpp_chat(
    *,
    host: str,
    model: str,
    messages: list[dict[str, str]],
    stop: list[str] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    response_schema: dict[str, Any] | None = None,
) -> str:
    url = f"{host.rstrip('/')}/v1/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if stop:
        payload["stop"] = stop
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if num_ctx is not None:
        payload["max_tokens"] = num_ctx
    if seed is not None:
        payload["seed"] = seed
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "rocky_response_v1",
                "strict": True,
                "schema": response_schema,
            },
        }
    parsed = _post_json(url, payload)
    choices = parsed.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise EndpointError("llama.cpp response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise EndpointError("llama.cpp choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise EndpointError("llama.cpp response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise EndpointError("llama.cpp response missing message.content")
    return content
