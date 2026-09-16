"""Validate and export manually authored data; never create conversation prose."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rocky.prompting import system_message

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/v1"
TRANSCRIPT = DATA / "source/film-transcript.txt"
SPLITS = {"train", "validation", "test", "reference"}
FOCUSES = {"film_world", "everyday", "embodiment"}
INPUTS = ("system.txt", "corpus/film.jsonl", "corpus/original.jsonl", "corpus/expansion.jsonl", "corpus/preferences.jsonl", "facts/cards.jsonl", "eval/probes.jsonl", "eval/trajectories.jsonl", "source/film-transcript.txt", "scenes.json")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(row)
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def expansion_rows(data: Path) -> list[dict]:
    """Expand handwritten Grace/Rocky pairs into roles, without authoring text."""
    result = []
    for row in read_jsonl(data / "corpus/expansion.jsonl"):
        if set(row) != {"id", "scene", "split", "focus", "context", "turns"}:
            raise ValueError("invalid expansion fields")
        if row["focus"] not in FOCUSES or row["split"] not in SPLITS - {"reference"}:
            raise ValueError(f"{row['id']}: invalid expansion focus or split")
        if not isinstance(row["turns"], list) or not row["turns"]:
            raise ValueError(f"{row['id']}: empty expansion turns")
        messages = []
        for pair in row["turns"]:
            if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(t, str) or not t.strip() for t in pair):
                raise ValueError(f"{row['id']}: expected nonempty Grace/Rocky pair")
            messages.extend(({"role": "user", "content": pair[0]}, {"role": "assistant", "content": pair[1]}))
        result.append({**{k: row[k] for k in ("id", "scene", "split", "context")}, "messages": messages})
    return result


def authored_rows(data: Path = DATA, curriculum: str = "expanded") -> list[dict]:
    if curriculum not in {"core", "expanded"}:
        raise ValueError("unknown curriculum")
    core = read_jsonl(data / "corpus/film.jsonl") + read_jsonl(data / "corpus/original.jsonl")
    return core + (expansion_rows(data) if curriculum == "expanded" else [])


def export_selections(data: Path) -> dict[str, list[dict]]:
    rows = authored_rows(data)
    selections = {split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "test")}
    # Both experiments use the same held-out validation/test sets. Only training changes.
    selections["core.train"] = [r for r in authored_rows(data, "core") if r["split"] == "train"]
    return selections


def preference_rows(data: Path = DATA) -> list[dict]:
    from rocky.tokenization import validate_messages

    rows = read_jsonl(data / "corpus/preferences.jsonl")
    instruction = (data / "system.txt").read_text(encoding="utf-8").strip()
    for row in rows:
        if set(row) != {"id", "scene", "split", "context", "messages", "chosen", "rejected", "criterion"}:
            raise ValueError("invalid preference fields")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("invalid preference split")
        for key in ("id", "scene", "context", "chosen", "rejected", "criterion"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("empty preference field")
        if normalized(row["chosen"]) == normalized(row["rejected"]):
            raise ValueError("preference alternatives must differ")
        prompt = [system_message(instruction, row["context"])] + row["messages"]
        validate_messages(prompt, prompt=True)
        for key in ("chosen", "rejected"):
            validate_messages(prompt + [{"role": "assistant", "content": row[key]}])
    return rows


def export_payloads(data: Path) -> tuple[dict, dict]:
    prompt = (data / "system.txt").read_text(encoding="utf-8").strip()
    payloads, index = {}, {}
    for split, selected in export_selections(data).items():
        payloads[split] = [{"messages": [system_message(prompt, r["context"])] + r["messages"]} for r in selected]
        index[split] = [r["id"] for r in selected]
    preferences = preference_rows(data)
    for split in ("train", "validation", "test"):
        selected = [r for r in preferences if r["split"] == split]
        name = "preferences." + split
        payloads[name] = [{
            "prompt": [system_message(prompt, r["context"])] + r["messages"],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        } for r in selected]
        index[name] = [r["id"] for r in selected]
    return payloads, index


def source_turns(path: Path) -> list[tuple[str, str]]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GRACE: "):
            turns.append(("user", line[7:]))
        elif line.startswith("ROCKY: "):
            turns.append(("assistant", line[7:]))
    return turns


def verify_transcription(row: dict, source: list[tuple[str, str]]) -> None:
    """Match a manually copied excerpt without replacing or correcting any text."""
    cursor = 0
    previous_source = None
    for message in row["messages"]:
        for line_index, line in enumerate(message["content"].splitlines()):
            pair = (message["role"], line)
            try:
                position = source.index(pair, cursor)
            except ValueError as error:
                raise ValueError(f"{row['id']}: text/speaker/order differs from transcript: {line!r}") from error
            if line_index and position != previous_source + 1:
                raise ValueError(f"{row['id']}: joined nonconsecutive source turns")
            cursor = position + 1
            previous_source = position


def validate(data: Path = DATA, transcript: Path | None = None) -> dict:
    from rocky.tokenization import validate_messages

    transcript = transcript or data / "source/film-transcript.txt"
    film = read_jsonl(data / "corpus/film.jsonl")
    original = read_jsonl(data / "corpus/original.jsonl")
    expansion = expansion_rows(data)
    facts = read_jsonl(data / "facts/cards.jsonl")
    probes = read_jsonl(data / "eval/probes.jsonl")
    trajectories_eval = read_jsonl(data / "eval/trajectories.jsonl")
    source = source_turns(transcript)
    ids = set()
    scenes = defaultdict(set)
    trajectories = set()
    split_counts = defaultdict(Counter)
    substantive = defaultdict(set)
    film_used = Counter()
    fields = {"id", "scene", "split", "context", "messages"}
    for kind, rows in (("film", film), ("original", original), ("expansion", expansion)):
        for row in rows:
            if set(row) != fields or row["split"] not in SPLITS:
                raise ValueError(f"invalid conversation fields: {row.get('id')}")
            if any(not isinstance(row[k], str) or not row[k].strip() for k in ("id", "scene", "context")):
                raise ValueError("empty conversation identity or context")
            if row["id"] in ids:
                raise ValueError(f"duplicate ID: {row['id']}")
            ids.add(row["id"])
            messages = row["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{row['id']}: empty messages")
            validate_messages([{"role": "system", "content": row["context"]}] + messages)
            previous = None
            for message in messages:
                if set(message) != {"role", "content"} or message["role"] not in {"user", "assistant"}:
                    raise ValueError(f"{row['id']}: invalid message")
                if not isinstance(message["content"], str) or not message["content"].strip():
                    raise ValueError(f"{row['id']}: empty content")
                if previous == message["role"]:
                    raise ValueError(f"{row['id']}: join consecutive same-speaker turns manually")
                previous = message["role"]
            if row["split"] != "reference":
                if messages[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in messages):
                    raise ValueError(f"{row['id']}: train/eval conversation needs Grace and final Rocky reply")
                scenes[row["scene"]].add(row["split"])
                trajectory = tuple((m["role"], normalized(m["content"])) for m in messages)
                if trajectory in trajectories:
                    raise ValueError(f"{row['id']}: duplicate conversation")
                trajectories.add(trajectory)
                # Long identical turns are leakage candidates; short natural replies are not.
                for message in messages:
                    for line in message["content"].splitlines():
                        if len(normalized(line).split()) >= 12:
                            substantive[(message["role"], normalized(line))].add(row["split"])
            if kind == "film":
                verify_transcription(row, source)
                film_used.update(line for m in messages if m["role"] == "assistant" for line in m["content"].splitlines())
            split_counts[row["split"]][kind + "_conversations"] += 1
            split_counts[row["split"]][kind + "_turns"] += sum(m["role"] == "assistant" for m in messages)
            split_counts[row["split"]][kind + "_words"] += sum(len(m["content"].split()) for m in messages if m["role"] == "assistant")
    if any(len(splits) > 1 for splits in scenes.values()):
        raise ValueError("a scene crosses train/validation/test boundaries")
    if any(len(splits) > 1 for splits in substantive.values()):
        raise ValueError("a substantive verbatim turn crosses split boundaries")
    preferences = preference_rows(data)
    preference_counts = Counter()
    preference_prompts = set()
    for row in preferences:
        if row["id"] in ids:
            raise ValueError("duplicate preference ID")
        ids.add(row["id"])
        scenes[row["scene"]].add(row["split"])
        signature = tuple(normalized(m["content"]) for m in row["messages"])
        if signature in preference_prompts:
            raise ValueError("duplicate preference prompt")
        preference_prompts.add(signature)
        preference_counts[row["split"]] += 1
        for other in film + original + expansion:
            if row["split"] != other["split"] and other["split"] != "reference":
                if any(normalized(m["content"]) == normalized(row["messages"][-1]["content"]) for m in other["messages"] if m["role"] == "user"):
                    raise ValueError("preference prompt crosses SFT split boundary")
    if any(len(splits) > 1 for splits in scenes.values()):
        raise ValueError("a preference scene crosses train/validation/test boundaries")
    # Preference history and BOTH alternatives must not carry long held-out text.
    for row in preferences:
        for message in row["messages"] + [{"role": "assistant", "content": row[key]} for key in ("chosen", "rejected")]:
            for line in message["content"].splitlines():
                if len(normalized(line).split()) >= 12:
                    substantive[(message["role"], normalized(line))].add(row["split"])
    if any(len(splits) > 1 for splits in substantive.values()):
        raise ValueError("substantive preference text crosses split boundaries")
    source_rocky = Counter(text for role, text in source if role == "assistant")
    if film_used - source_rocky:
        raise ValueError("a source Rocky turn is copied more often than it occurs")
    missing = source_rocky - film_used
    if any("[Eridian" not in text for text in missing):
        raise ValueError(f"unaccounted translated source turns: {list(missing)}")
    for card in facts:
        if set(card) != {"id", "fact", "scope", "available", "verified"}:
            raise ValueError("invalid fact card fields")
        if type(card["verified"]) is not bool or card["scope"] not in {"film", "project"}:
            raise ValueError("invalid fact verification or scope")
        if any(not isinstance(card[k], str) or not card[k].strip() for k in ("id", "fact", "available")):
            raise ValueError("empty fact")
        if card["id"] in ids:
            raise ValueError("duplicate fact ID")
        ids.add(card["id"])
    for row in trajectories_eval:
        if set(row) != {"id", "scene", "kind", "context", "steps"} or not isinstance(row["steps"], list) or len(row["steps"]) < 2:
            raise ValueError("invalid evaluation trajectory")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ("id", "scene", "kind", "context")):
            raise ValueError("invalid trajectory metadata")
        if row["id"] in ids:
            raise ValueError("duplicate trajectory ID")
        ids.add(row["id"])
        for step in row["steps"]:
            if set(step) != {"prompt", "must", "must_not"}:
                raise ValueError("invalid trajectory step")
    checks = probes + [{**{k: r[k] for k in ("scene", "context", "kind")}, **step,
                        "id": r["id"] + f"-step-{i}"} for r in trajectories_eval for i, step in enumerate(r["steps"])]
    for probe in checks:
        if set(probe) != {"id", "scene", "context", "prompt", "must", "must_not", "kind"}:
            raise ValueError("invalid evaluation probe")
        if probe["id"] in ids:
            raise ValueError("duplicate probe ID")
        ids.add(probe["id"])
        if not probe["must"] or not probe["must_not"]:
            raise ValueError("probe needs positive and negative criteria")
        if any(not isinstance(probe[k], str) or not probe[k].strip() for k in ("id", "scene", "context", "prompt", "kind")):
            raise ValueError("empty evaluation field")
        if any(not isinstance(probe[k], list) or any(not isinstance(t, str) or not t.strip() for t in probe[k]) for k in ("must", "must_not")):
            raise ValueError("invalid evaluation criteria")
        validate_messages([system_message("Rocky", probe["context"]), {"role": "user", "content": probe["prompt"]}], prompt=True)
        for row in film + original + expansion:
            if row["split"] == "train" and any(normalized(m["content"]) == normalized(probe["prompt"]) for m in row["messages"] if m["role"] == "user"):
                raise ValueError(f"{probe['id']}: exact training prompt leakage")
        for row in preferences:
            if row["split"] == "train" and any(normalized(m["content"]) == normalized(probe["prompt"]) for m in row["messages"] if m["role"] == "user"):
                raise ValueError(f"{probe['id']}: preference training prompt leakage")
    from rocky.session import SceneSession

    SceneSession(data=data)
    train = split_counts["train"]
    if train["film_turns"] <= train["original_turns"] * 3 or train["film_words"] <= train["original_words"] * 3:
        raise ValueError("film must exceed 75% of core target turns and words")
    focus_counts = defaultdict(Counter)
    for row in read_jsonl(data / "corpus/expansion.jsonl"):
        focus_counts[row["split"]][row["focus"] + "_conversations"] += 1
        focus_counts[row["split"]][row["focus"] + "_turns"] += len(row["turns"])
    totals = {split: {
        "conversations": sum(counts[k] for k in counts if k.endswith("_conversations")),
        "assistant_turns": sum(counts[k] for k in counts if k.endswith("_turns")),
        "assistant_words": sum(counts[k] for k in counts if k.endswith("_words")),
    } for split, counts in split_counts.items()}
    lengths = [len(text.split()) for role, text in source if role == "assistant"]
    return {
        "schema": "rocky-persona-v1",
        "splits": dict(split_counts),
        "totals": totals,
        "expansion_focus": dict(focus_counts),
        "fact_cards": len(facts),
        "preferences": dict(preference_counts),
        "evaluation_probes": len(probes),
        "evaluation_trajectories": len(trajectories_eval),
        "evaluation_trajectory_turns": sum(len(r["steps"]) for r in trajectories_eval),
        "source_rocky_turns": sum(source_rocky.values()),
        "represented_source_turns": sum(film_used.values()),
        "untranslated_source_turns": sum(missing.values()),
        "source_words_per_turn": {"mean": round(statistics.mean(lengths), 2), "median": statistics.median(lengths)},
        "checks": {"transcription": True, "scene_disjoint": True, "exact_leakage": True},
        "limitations": ["No independent film/audio verification", "Semantic leakage requires review", "No trained-model evaluation performed"],
    }


def export(data: Path = DATA, transcript: Path | None = None) -> dict:
    transcript = transcript or data / "source/film-transcript.txt"
    report = validate(data, transcript)
    output = data / "exports"
    output.mkdir(exist_ok=True)
    payloads, index = export_payloads(data)
    for split, formatted in payloads.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in formatted), encoding="utf-8")
    inputs = [data / name for name in INPUTS]
    manifest = {
        **report,
        "status": "prepared_for_local_experiment",
        "model_evaluated": False,
        "inputs": {str(path.relative_to(data)): digest(path) for path in inputs},
        "source_sha256": digest(transcript),
        "exports": {f"{split}.jsonl": digest(output / f"{split}.jsonl") for split in index},
        "row_ids": index,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_export(data: Path = DATA, transcript: Path | None = None) -> dict:
    transcript = transcript or data / "source/film-transcript.txt"
    report = validate(data, transcript)
    output = data / "exports"
    manifest = json.loads((output / "manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in report.items()):
        raise ValueError("manifest inventory differs from current validated data; rebuild export")
    payloads, index = export_payloads(data)
    if set(manifest["inputs"]) != set(INPUTS) or set(manifest["exports"]) != {name + ".jsonl" for name in payloads}:
        raise ValueError("manifest omits required inputs or exports; rebuild export")
    if manifest["source_sha256"] != digest(transcript):
        raise ValueError("source transcript changed; review and rebuild")
    for path, expected in manifest["inputs"].items():
        if digest(data / path) != expected:
            raise ValueError(f"input changed: {path}; rebuild export")
    for path, expected in manifest["exports"].items():
        if digest(output / path) != expected:
            raise ValueError(f"export changed: {path}; rebuild export")
    for split, expected in payloads.items():
        if read_jsonl(output / f"{split}.jsonl") != expected:
            raise ValueError(f"{split}: export does not exactly preserve authored conversations")
        if manifest["row_ids"][split] != index[split]:
            raise ValueError("manifest row order differs")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    action = {"check": validate, "export": export, "verify": verify_export}[args.command]
    print(json.dumps(action(), indent=2))


if __name__ == "__main__":
    main()
