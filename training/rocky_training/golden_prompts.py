from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldenPrompt:
    id: str
    scenario_family: str
    user: str
    quality_focus: str | None = None
    humor_expectation: str | None = None
    grounding_notes: str | None = None
    grounding_patterns: tuple[str, ...] = ()
    uncertainty_patterns: tuple[str, ...] = ()
    roleplay_forbidden_patterns: tuple[str, ...] = ()
    book_fact_forbidden_patterns: tuple[str, ...] = ()


class GoldenPromptError(Exception):
    pass


def _read_optional_string_tuple(
    parsed: dict[str, object], field: str, line_number: int
) -> tuple[str, ...]:
    raw = parsed.get(field)
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(entry, str) or not entry for entry in raw):
        raise GoldenPromptError(f"line {line_number}: {field} must be a string array")
    return tuple(raw)


def load_golden_prompts(path: str | Path, *, limit: int = 0) -> list[GoldenPrompt]:
    prompts: list[GoldenPrompt] = []
    file_path = Path(path)

    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise GoldenPromptError(f"line {line_number}: invalid json") from error

        if not isinstance(parsed, dict):
            raise GoldenPromptError(f"line {line_number}: row must be an object")

        prompt_id = parsed.get("id")
        scenario_family = parsed.get("scenarioFamily")
        user = parsed.get("user")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise GoldenPromptError(f"line {line_number}: id must be a non-empty string")
        if not isinstance(scenario_family, str) or not scenario_family:
            raise GoldenPromptError(
                f"line {line_number}: scenarioFamily must be a non-empty string"
            )
        if not isinstance(user, str) or not user:
            raise GoldenPromptError(f"line {line_number}: user must be a non-empty string")

        quality_focus_raw = parsed.get("qualityFocus")
        quality_focus: str | None
        if quality_focus_raw is None:
            quality_focus = None
        elif isinstance(quality_focus_raw, str) and quality_focus_raw.strip():
            quality_focus = quality_focus_raw
        else:
            raise GoldenPromptError(
                f"line {line_number}: qualityFocus must be a non-empty string when present"
            )

        humor_expectation_raw = parsed.get("humorExpectation")
        humor_expectation: str | None
        if humor_expectation_raw is None:
            humor_expectation = None
        elif humor_expectation_raw in {"expected", "optional", "avoid"}:
            humor_expectation = str(humor_expectation_raw)
        else:
            raise GoldenPromptError(
                f"line {line_number}: humorExpectation must be expected, optional, or avoid"
            )

        grounding_notes_raw = parsed.get("groundingNotes")
        grounding_notes: str | None
        if grounding_notes_raw is None:
            grounding_notes = None
        elif isinstance(grounding_notes_raw, str) and grounding_notes_raw.strip():
            grounding_notes = grounding_notes_raw
        else:
            raise GoldenPromptError(
                f"line {line_number}: groundingNotes must be a non-empty string when present"
            )

        prompts.append(
            GoldenPrompt(
                id=prompt_id,
                scenario_family=scenario_family,
                user=user,
                quality_focus=quality_focus,
                humor_expectation=humor_expectation,
                grounding_notes=grounding_notes,
                grounding_patterns=_read_optional_string_tuple(
                    parsed, "groundingPatterns", line_number
                ),
                uncertainty_patterns=_read_optional_string_tuple(
                    parsed, "uncertaintyPatterns", line_number
                ),
                roleplay_forbidden_patterns=_read_optional_string_tuple(
                    parsed, "roleplayForbiddenPatterns", line_number
                ),
                book_fact_forbidden_patterns=_read_optional_string_tuple(
                    parsed, "bookFactForbiddenPatterns", line_number
                ),
            )
        )
        if limit > 0 and len(prompts) >= limit:
            break

    if len(prompts) == 0:
        raise GoldenPromptError("golden prompt file contains no rows")
    return prompts
