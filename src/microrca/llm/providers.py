"""Provider-agnostic chat interface with tool calling.

Supported providers (chosen with --provider):
  openai     OpenAI API (OPENAI_API_KEY). Also any OpenAI-compatible endpoint via
             OPENAI_BASE_URL, e.g. Groq, Together, OpenRouter, vLLM, Ollama (http://localhost:11434/v1).
  anthropic  Anthropic Messages API (ANTHROPIC_API_KEY).
  gemini     Google Gemini through its OpenAI-compatible endpoint (GEMINI_API_KEY).
  scripted   Offline deterministic policy. It is NOT a language model: it exists so the
             full pipeline (state, tool calls, output contract, evaluation) runs in CI and
             without any API key. Never report its numbers as LLM results.

All providers use only `httpx`, so no vendor SDK is required.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"id", "name", "arguments"}]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class ChatModel:
    name = "base"

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        raise NotImplementedError


class OpenAICompatible(ChatModel):
    """OpenAI chat-completions format; works for OpenAI, Gemini, Groq, Ollama, vLLM..."""

    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.0):
        self.name = model
        self.model, self.base_url, self.api_key, self.temperature = model, base_url.rstrip("/"), api_key, temperature

    def chat(self, messages, tools=None):
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        t0 = time.time()
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=120,
                       headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = [
            {"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
            for c in msg.get("tool_calls") or []
        ]
        usage = data.get("usage", {})
        return LLMReply(msg.get("content") or "", calls, usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0), time.time() - t0)


class Anthropic(ChatModel):
    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        self.name = model
        self.model, self.api_key, self.temperature = model, api_key, temperature

    @staticmethod
    def _convert(messages):
        """OpenAI-style history -> Anthropic system + messages."""
        system, out = "", []
        for m in messages:
            if m["role"] == "system":
                system += m["content"]
            elif m["role"] == "tool":
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                            "input": json.loads(c["function"]["arguments"])} for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return system, out

    def chat(self, messages, tools=None):
        system, msgs = self._convert(messages)
        body: dict[str, Any] = {"model": self.model, "max_tokens": 2048, "system": system,
                                "messages": msgs, "temperature": self.temperature}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                             for t in tools]
        t0 = time.time()
        r = httpx.post("https://api.anthropic.com/v1/messages", json=body, timeout=120, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        data = r.json()
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        calls = [{"id": b["id"], "name": b["name"], "arguments": b["input"]}
                 for b in data["content"] if b["type"] == "tool_use"]
        u = data.get("usage", {})
        return LLMReply(text, calls, u.get("input_tokens", 0), u.get("output_tokens", 0), time.time() - t0)


class Scripted(ChatModel):
    """Deterministic stand-in for an LLM, used for offline smoke tests only.

    Tool-calling mode follows a fixed investigation script: metric anomalies, then
    logs and traces of the top suspects, then answers with the top metric suspect.
    """

    name = "scripted"

    def chat(self, messages, tools=None):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if tools and not tool_msgs:
            return LLMReply(tool_calls=[{"id": "c0", "name": "detect_metric_anomalies", "arguments": {"top_k": 5}}])
        source = tool_msgs[0]["content"] if tool_msgs else messages[-1]["content"]
        suspects = _suspects_from(source)
        if tools and len(tool_msgs) == 1 and suspects:
            calls = []
            for i, s in enumerate(suspects[:2]):
                calls += [{"id": f"l{i}", "name": "get_log_summary", "arguments": {"service": s}},
                          {"id": f"t{i}", "name": "get_trace_summary", "arguments": {"service": s}}]
            return LLMReply(tool_calls=calls)
        answer = {"ranking": [{"service": s, "fault_type": "unknown", "confidence": 0.5,
                               "evidence": ["highest metric anomaly score"]} for s in suspects[:5]],
                  "summary": "scripted policy answer"}
        return LLMReply(text=json.dumps(answer))


def _suspects_from(text: str) -> list[str]:
    key = text.find('"top_anomalies"')
    if key < 0:
        return []
    rows = json.JSONDecoder().raw_decode(text[text.find("[", key):])[0]
    seen: list[str] = []
    for r in rows:
        if r["service"] not in seen:
            seen.append(r["service"])
    return seen


def get_model(provider: str, model: str | None = None, temperature: float = 0.0) -> ChatModel:
    if provider == "scripted":
        return Scripted()
    if provider == "openai":
        return OpenAICompatible(model or "gpt-4o-mini", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                os.environ.get("OPENAI_API_KEY", "none"), temperature)
    if provider == "gemini":
        return OpenAICompatible(model or "gemini-2.0-flash",
                                "https://generativelanguage.googleapis.com/v1beta/openai",
                                os.environ["GEMINI_API_KEY"], temperature)
    if provider == "anthropic":
        return Anthropic(model or "claude-haiku-4-5", os.environ["ANTHROPIC_API_KEY"], temperature)
    raise ValueError(f"unknown provider {provider}")
