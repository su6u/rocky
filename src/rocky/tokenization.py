"""Native Gemma prefill with explicit reply-only labels, including turn termination."""

from __future__ import annotations

from typing import Any

CONTROL_MARKERS = ("<|", "<turn|>", "<channel|>", "<bos>", "<eos>", "<pad>", "[end_of_turn]")


def validate_messages(messages: list[dict], *, prompt: bool = False) -> None:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("conversation must start with one system message")
    previous = "system"
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError("only role and content are permitted")
        role, content = message["role"], message["content"]
        if role not in ({"system"} if index == 0 else {"user", "assistant"}):
            raise ValueError("invalid role or embedded system message")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty message")
        if content != content.strip():
            raise ValueError("leading/trailing whitespace requires manual correction")
        if any(marker in content for marker in CONTROL_MARKERS):
            raise ValueError("reserved control token in conversation text")
        if index and previous == role:
            raise ValueError("consecutive same-speaker messages")
        previous = role
    if prompt and len(messages) > 1 and messages[-1]["role"] != "user":
        raise ValueError("generation prompt must end with Grace")


def render_prompt(tokenizer: Any, messages: list[dict]) -> str:
    validate_messages(messages, prompt=True)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_reply(tokenizer: Any, prompt: list[dict], reply: str, max_length: int) -> dict:
    validate_messages([*prompt, {"role": "assistant", "content": reply}])
    prefix = render_prompt(tokenizer, prompt)
    stop = "<turn|>"
    stop_id = tokenizer.convert_tokens_to_ids(stop)
    if stop_id is None or stop_id == tokenizer.unk_token_id or tokenizer.encode(stop, add_special_tokens=False) != [stop_id]:
        raise ValueError("Gemma turn terminator must be a recognized single token")
    rendered = prefix + reply + stop
    # Use this checkpoint's non-thinking prefix. E4B does not use the larger
    # variants' empty thought block. Historical rendering must preserve speech.
    native = tokenizer.apply_chat_template(
        [*prompt, {"role": "assistant", "content": reply}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if not native.endswith(reply + stop + "\n"):
        raise ValueError("native template changes reply text or turn termination")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_length:
        raise ValueError(f"{len(ids)} tokens exceeds max_length={max_length}; no silent truncation")
    if len(ids) != len(offsets) or not ids:
        raise ValueError("missing tokenizer character offsets")
    labels = []
    content_tokens = 0
    for token_id, (start, end) in zip(ids, offsets):
        if start < len(prefix) < end:
            raise ValueError("token crosses prompt/reply boundary")
        target = start >= len(prefix) and end > start
        labels.append(token_id if target else -100)
        content_tokens += int(target and start < len(prefix) + len(reply))
    if not content_tokens or ids[-1] != stop_id or labels[-1] != stop_id or labels[0] != -100:
        raise ValueError("reply content and end-of-turn must be supervised; prefix must be masked")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def encode_conversations(tokenizer: Any, rows: list[dict], max_length: int) -> tuple[list[dict], dict]:
    encoded = []
    for row in rows:
        messages = row["messages"]
        validate_messages(messages)
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                # Each authored reply is supervised once with its native inference prefix.
                # These are in-memory loss units, not newly authored/exported conversations.
                encoded.append(encode_reply(tokenizer, messages[:index], message["content"], max_length))
    if not encoded:
        raise ValueError("no supervised replies")
    return encoded, {
        "conversations": len(rows), "supervised_replies": len(encoded),
        "input_tokens": sum(len(row["input_ids"]) for row in encoded),
        "target_tokens": sum(sum(t != -100 for t in row["labels"]) for row in encoded),
        "max_tokens": max(len(row["input_ids"]) for row in encoded),
    }


class ReplyCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows: list[dict]) -> dict:
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([
            row[key] + [padding] * (width - len(row[key])) for row in rows
        ], dtype=torch.long) for key, padding in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100))}
