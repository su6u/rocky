"""Read-only corpus diagnostics; no authored text or test answers are exported here."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from rocky.data import DATA, authored_rows, normalized, read_jsonl, verify_export


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {"count": len(values), "min": ordered[0], "max": ordered[-1],
            "mean": round(sum(values) / len(values), 3),
            **{f"p{p}": ordered[max(0, math.ceil(len(values) * p / 100) - 1)] for p in (50, 90, 95, 99)}}


def token_diagnostics(rows: list[dict]) -> dict:
    units = [unit for row in rows for unit in ([row["chosen"], row["rejected"]] if "chosen" in row else [row])]
    inputs = [len(row["input_ids"]) for row in units]
    targets = [sum(t != -100 for t in row["labels"]) for row in units]
    return {"input_length": distribution(inputs), "target_length": distribution(targets),
            "supervised_fraction": sum(targets) / sum(inputs) if inputs else 0,
            "input_tokens": sum(inputs), "target_tokens": sum(targets)}


def near_overlap(rows: list[dict], threshold: float = 0.35) -> list[dict]:
    """Flag shared five-word phrases across splits; a review aid, not semantic proof."""
    def shingles(row):
        result = set()
        for message in row["messages"]:
            words = normalized(message["content"]).split()
            result.update(tuple(words[i:i + 5]) for i in range(len(words) - 4))
        return result

    train = [(row, shingles(row)) for row in rows if row["split"] == "train"]
    flagged = []
    for held in (row for row in rows if row["split"] in {"validation", "test"}):
        right = shingles(held)
        for row, left in train:
            if len(left) < 4 or len(right) < 4:
                continue
            overlap = len(left & right) / min(len(left), len(right))
            if overlap >= threshold:
                flagged.append({"train_id": row["id"], "heldout_id": held["id"],
                                "split": held["split"], "fivegram_containment": round(overlap, 3)})
    return sorted(flagged, key=lambda row: (-row["fivegram_containment"], row["heldout_id"], row["train_id"]))


def dataset_analysis(data: Path = DATA) -> dict:
    inventory = verify_export(data)
    rows = authored_rows(data)
    groups = []
    for split in ("train", "validation", "test"):
        for kind, prefix in (("film", "film-"), ("original", "original-"), ("expansion", None)):
            selected = [row for row in rows if row["split"] == split and
                        (row["id"].startswith(prefix) if prefix else
                         not row["id"].startswith(("film-", "original-")))]
            lengths = [len(m["content"].split()) for row in selected for m in row["messages"]
                       if m["role"] == "assistant"]
            groups.append({"split": split, "source": kind, "conversations": len(selected),
                           "replies": len(lengths), "target_words": sum(lengths),
                           "reply_words": distribution(lengths)})
    train_counts = inventory["splits"]["train"]
    mixes = {}
    for curriculum, kinds in (("core", ("film", "original")), ("expanded", ("film", "original", "expansion"))):
        mixes[curriculum] = {unit: train_counts[f"film_{unit}"] /
                            sum(train_counts[f"{kind}_{unit}"] for kind in kinds) for unit in ("turns", "words")}
    facts = read_jsonl(data / "facts/cards.jsonl")
    import json

    scenes = json.loads((data / "scenes.json").read_text())
    used = {key for scene in scenes.values() for key in scene["facts"]}
    return {"inventory": inventory, "groups": groups, "film_fraction": mixes,
            "facts_by_phase": dict(Counter(card["available"] for card in facts)),
            "unverified_fact_ids": [card["id"] for card in facts if not card["verified"]],
            "unmapped_fact_ids": [card["id"] for card in facts if card["id"] not in used],
            "near_overlap_candidates": near_overlap(rows),
            "limitations": ["Five-word overlap is lexical only; no semantic certification",
                            "Fact cards are context resources, not automatically supervised replies",
                            "Expanded curriculum is predominantly original fiction; compare against core"]}
