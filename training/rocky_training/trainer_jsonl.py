"""load deterministic JSONL trainer exports and artifact manifests"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rocky_training.eval_gates import response_is_valid


@dataclass(frozen=True)
class TrainerExportMessage:
    role: str
    content: str


@dataclass(frozen=True)
class TrainerExportRow:
    id: str
    messages: tuple[TrainerExportMessage, ...]


class TrainerJsonlError(Exception):
    pass


def load_trainer_jsonl(path: str | Path, *, max_rows: int | None = None) -> list[TrainerExportRow]:
    rows: list[TrainerExportRow] = []
    file_path = Path(path)

    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainerJsonlError(f"line {line_number}: invalid json") from error

        if not isinstance(parsed, dict):
            raise TrainerJsonlError(f"line {line_number}: row must be an object")
        row_id = parsed.get("id")
        messages_raw = parsed.get("messages")
        if not isinstance(row_id, str) or not row_id:
            raise TrainerJsonlError(f"line {line_number}: id must be a non-empty string")
        if not isinstance(messages_raw, list) or len(messages_raw) == 0:
            raise TrainerJsonlError(f"line {line_number}: messages must be a non-empty array")

        messages: list[TrainerExportMessage] = []
        for index, message in enumerate(messages_raw):
            if not isinstance(message, dict):
                raise TrainerJsonlError(f"line {line_number}: messages[{index}] must be an object")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise TrainerJsonlError(
                    f"line {line_number}: messages[{index}].role must be a non-empty string"
                )
            if not isinstance(content, str) or not content:
                raise TrainerJsonlError(
                    f"line {line_number}: messages[{index}].content must be a non-empty string"
                )
            if role == "assistant" and not response_is_valid(content):
                raise TrainerJsonlError(
                    f"line {line_number}: messages[{index}].content must be a valid Rocky v1 response"
                )
            messages.append(TrainerExportMessage(role=role, content=content))

        rows.append(TrainerExportRow(id=row_id, messages=tuple(messages)))
        if max_rows is not None and max_rows > 0 and len(rows) >= max_rows:
            break

    if len(rows) == 0:
        raise TrainerJsonlError("dataset contains no rows")
    return rows


def format_smoke_example(row: TrainerExportRow) -> tuple[str, str]:
    system = next((message.content for message in row.messages if message.role == "system"), "")
    user = next((message.content for message in row.messages if message.role == "user"), "")
    assistant = next((message.content for message in row.messages if message.role == "assistant"), "")
    prompt = f"{system}\nUser: {user}\nAssistant:"
    return prompt, assistant


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
