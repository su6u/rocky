"""Scene-bounded prompt assembly and exact playback; no model or motor side effects."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from rocky.data import DATA, read_jsonl
from rocky.inference import format_failure
from rocky.prompting import system_message
from rocky.tokenization import validate_messages


class SceneSession:
    def __init__(self, scene: str = "survivors", *, data: Path = DATA):
        self.data = data
        self.scenes = json.loads((data / "scenes.json").read_text())
        self.cards = {card["id"]: card for card in read_jsonl(data / "facts/cards.jsonl")}
        for spec in self.scenes.values():
            if set(spec) != {"context", "facts"} or len(spec["facts"]) != len(set(spec["facts"])):
                raise ValueError("invalid scene specification")
            if any(key not in self.cards for key in spec["facts"]):
                raise ValueError("unknown scene fact")
        self.reset(scene)

    def reset(self, scene: str) -> None:
        if scene not in self.scenes:
            raise ValueError("unknown scene; choose an explicitly authored scene")
        self.scene = scene
        self.history = []

    def prompt(self, grace: str) -> list[dict]:
        spec = self.scenes[self.scene]
        facts = [self.cards[key]["fact"] for key in spec["facts"] if self.cards[key]["verified"]]
        system = system_message((self.data / "system.txt").read_text(), spec["context"], facts)
        messages = [system] + deepcopy(self.history)
        messages.append({"role": "user", "content": grace})
        validate_messages(messages, prompt=True)
        return messages

    def accept(self, grace: str, rocky: str) -> None:
        if format_failure(rocky):
            raise ValueError("model returned metadata or another speaker; history unchanged")
        messages = self.prompt(grace) + [{"role": "assistant", "content": rocky}]
        validate_messages(messages)
        self.history.extend(deepcopy(messages[-2:]))


class FilmPlayback:
    """Exact script follower. A mismatch never advances the cursor or fabricates a reply."""

    def __init__(self, record: str, *, data: Path = DATA):
        rows = {row["id"]: row for row in read_jsonl(data / "corpus/film.jsonl")}
        if record not in rows or rows[record]["split"] == "reference":
            raise ValueError("choose a Grace/Rocky film conversation")
        self.messages = deepcopy(rows[record]["messages"])
        self.cursor = 0

    def opening(self) -> str | None:
        if self.cursor == 0 and self.messages[0]["role"] == "assistant":
            self.cursor = 1
            return self.messages[0]["content"]
        return None

    def reply(self, grace: str) -> str:
        if self.cursor >= len(self.messages):
            raise ValueError("scene has ended")
        expected = self.messages[self.cursor]
        if expected["role"] != "user":
            raise ValueError("read Rocky's opening first")
        if " ".join(grace.split()) != " ".join(expected["content"].split()):
            raise ValueError("Grace's line differs from the selected script; cursor unchanged")
        response = self.messages[self.cursor + 1]["content"]
        self.cursor += 2
        return response
