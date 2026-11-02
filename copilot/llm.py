"""Thin LLM client abstraction.

The engine depends on the `LLMClient` protocol, not on OpenAI directly, so
tests can use a scripted fake and other providers can be plugged in.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMResponse:
    data: dict
    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_call: list[str] = field(default_factory=list)

    def add(self, resp: LLMResponse, label: str) -> None:
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.calls += 1
        self.by_call.append(label)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    model: str

    def complete_json(self, messages: list[dict]) -> LLMResponse: ...


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, tolerating markdown fences or leading prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class OpenAIClient:
    """OpenAI Chat Completions in JSON mode."""

    def __init__(self, model: str, api_key: str | None = None, timeout_s: float = 60.0) -> None:
        from openai import OpenAI  # imported lazily so tests don't need the package

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout_s)

    def complete_json(self, messages: list[dict]) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            data=parse_json_object(content),
            raw=content,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
