"""Bounded text inference; no hidden retries, history truncation, or motor execution."""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from rocky.prompting import check_budget
from rocky.tokenization import CONTROL_MARKERS


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 128
    context_tokens: int = 4096

    def __post_init__(self):
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("invalid generation temperature")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("invalid top_p")
        if type(self.context_tokens) is not int or type(self.max_tokens) is not int or not 0 < self.max_tokens < self.context_tokens <= 131072:
            raise ValueError("invalid E4B context/reply budget")

    def record(self):
        return asdict(self)


def format_failure(content: str) -> bool:
    return (any(marker in content for marker in CONTROL_MARKERS)
            or content.lstrip().startswith(("{", "Rocky:", "ROCKY:", "Grace:", "GRACE:")))


class InferenceClient:
    def __init__(self, endpoint: str, model: str, tokenizer, settings: GenerationSettings | None = None):
        if urlparse(endpoint).scheme not in {"http", "https"} or not urlparse(endpoint).netloc:
            raise ValueError("endpoint must be an explicit HTTP(S) inference server")
        if not model.strip():
            raise ValueError("served model name is required")
        self.endpoint, self.model, self.tokenizer = endpoint, model, tokenizer
        self.settings = settings or GenerationSettings()

    def complete(self, messages: list[dict], *, timeout: int = 60) -> str:
        check_budget(self.tokenizer, messages, self.settings.context_tokens, self.settings.max_tokens)
        body = {"model": self.model, "messages": messages, "temperature": self.settings.temperature,
                "top_p": self.settings.top_p, "max_tokens": self.settings.max_tokens,
                "stop": ["<turn|>"], "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(self.endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("inference server returned an invalid completion envelope") from error
        if not isinstance(content, str) or not content.strip():
            raise ValueError("inference server returned no spoken text")
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"completion did not finish normally: {choice.get('finish_reason')}")
        return content.strip()


def load_tokenizer():
    from transformers import AutoTokenizer

    from rocky.config import load_config, tokenizer_location

    config = load_config()
    tokenizer_source, tokenizer_kwargs = tokenizer_location(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, **tokenizer_kwargs)
    if not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str):
        raise ValueError("pinned native E4B tokenizer required")
    return tokenizer
