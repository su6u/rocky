"""parse exactly one Rocky v1 response object"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


RESPONSE_FIELDS = {"spoken", "emotion", "intensity", "gesture", "callbackId"}


@dataclass(frozen=True)
class ParsedModelOutput:
    spoken: str
    response_json: str | None


def parse_response_object(raw_output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != RESPONSE_FIELDS:
        return None
    return parsed


def parse_model_output(raw_output: str) -> ParsedModelOutput:
    response = parse_response_object(raw_output)
    if response is None:
        return ParsedModelOutput(spoken=raw_output.strip(), response_json=None)

    spoken = response.get("spoken")
    if not isinstance(spoken, str):
        return ParsedModelOutput(spoken=raw_output.strip(), response_json=None)

    return ParsedModelOutput(
        spoken=spoken.strip(),
        response_json=json.dumps(response, ensure_ascii=False, separators=(",", ":")),
    )


def serialize_response(value: object) -> str:
    if not isinstance(value, dict) or set(value) != RESPONSE_FIELDS:
        raise ValueError("response must contain exactly the Rocky v1 response fields")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def slugify_label(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "model"
