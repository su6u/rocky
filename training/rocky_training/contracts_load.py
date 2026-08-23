"""load canonical generated contracts shared with the TypeScript runtime"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from rocky_training.paths import default_contracts_dir


class ContractsError(Exception):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractsError(f"contracts file missing: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractsError(f"invalid json in {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise ContractsError(f"contracts file must be an object: {path}")
    return parsed


@lru_cache(maxsize=1)
def load_domain_contract(contracts_dir: str | None = None) -> dict[str, Any]:
    root = Path(contracts_dir) if contracts_dir is not None else default_contracts_dir()
    data = _read_json(root / "domain.json")
    emotions = data.get("emotions")
    gestures = data.get("gestures")
    if not isinstance(emotions, list) or not all(isinstance(item, str) for item in emotions):
        raise ContractsError("domain.json emotions must be a string array")
    if not isinstance(gestures, list) or not all(isinstance(item, str) for item in gestures):
        raise ContractsError("domain.json gestures must be a string array")
    return {
        "emotions": frozenset(emotions),
        "gestures": frozenset(gestures),
    }


@lru_cache(maxsize=1)
def load_eval_gate_phrases(contracts_dir: str | None = None) -> dict[str, tuple[str, ...]]:
    root = Path(contracts_dir) if contracts_dir is not None else default_contracts_dir()
    data = _read_json(root / "eval-gates.json")
    fields = (
        "promptInjectionPhrases",
        "bookFactTrapPhrases",
        "assistantRegisterPhrases",
        "thinkingLeakPhrases",
        "thirdPersonGracePatterns",
    )
    result: dict[str, tuple[str, ...]] = {}
    for field in fields:
        value = data.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ContractsError(f"eval-gates.json {field} must be a string array")
        result[field] = tuple(value)
    return result


@lru_cache(maxsize=1)
def load_protocol_contract(contracts_dir: str | None = None) -> dict[str, str]:
    root = Path(contracts_dir) if contracts_dir is not None else default_contracts_dir()
    data = _read_json(root / "protocol.json")
    preamble = data.get("contextPreamble")
    if not isinstance(preamble, str) or not preamble:
        raise ContractsError("protocol.json contextPreamble must be a non-empty string")
    return {"context_preamble": preamble}
