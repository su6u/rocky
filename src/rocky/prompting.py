"""One system-message format and explicit native-token context accounting."""

from __future__ import annotations

from rocky.tokenization import render_prompt


def system_message(instruction: str, context: str, facts=()) -> dict:
    content = instruction.strip() + "\n\nScene: " + context
    if facts:
        content += "\n\nEstablished information:\n" + "\n".join(facts)
    return {"role": "system", "content": content}


def check_budget(tokenizer, messages: list[dict], context_tokens: int, reply_tokens: int) -> int:
    if type(context_tokens) is not int or type(reply_tokens) is not int or not 0 < reply_tokens < context_tokens:
        raise ValueError("invalid context/reply token budget")
    rendered = render_prompt(tokenizer, messages)
    count = len(tokenizer.encode(rendered, add_special_tokens=False))
    if count + reply_tokens > context_tokens:
        raise ValueError(f"context budget exceeded ({count} input + {reply_tokens} reserved > {context_tokens}); "
                         "history was not dropped; start a fresh scene or explicitly increase the context budget")
    return count
