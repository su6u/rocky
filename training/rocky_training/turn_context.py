from __future__ import annotations

from rocky_training.contracts_load import load_protocol_contract


def clean_grounding_notes(grounding_notes: str | None) -> str | None:
    if grounding_notes is None:
        return None
    cleaned = grounding_notes.strip()
    return cleaned if cleaned else None


def build_context_message(
    *,
    grounding_notes: str | None = None,
    memory_facts: list[str] | None = None,
) -> dict[str, str] | None:
    """Mirror @rocky/protocol buildContextMessage for serving-parity eval."""
    notes = clean_grounding_notes(grounding_notes)
    facts = [fact.replace("\n", " ").strip() for fact in memory_facts or [] if fact.strip()]
    sections: list[str] = []
    if facts:
        sections.append("Memory facts:\n" + "\n".join(f"- {fact}" for fact in facts))
    if notes:
        sections.append(f"Grounding notes:\n{notes}")
    if not sections:
        return None
    preamble = load_protocol_contract()["context_preamble"]
    return {
        "role": "user",
        "content": f"{preamble}\n\n" + "\n\n".join(sections),
    }


def prepare_turn_messages(
    messages: list[dict[str, str]],
    *,
    grounding_notes: str | None = None,
    memory_facts: list[str] | None = None,
) -> list[dict[str, str]]:
    context = build_context_message(grounding_notes=grounding_notes, memory_facts=memory_facts)
    return [context, *messages] if context is not None else messages
