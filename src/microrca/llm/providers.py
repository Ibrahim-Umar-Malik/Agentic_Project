"""Provider-agnostic chat interface with tool calling, retry handling, and .env loading.

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

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


def load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load key=value pairs from .env if present without overriding existing environment variables."""
    path = dotenv_path or (Path(__file__).resolve().parents[3] / ".env")
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"id", "name", "arguments"}]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    retries: int = 0


class ChatModel:
    name = "base"
    seed: int = 42
    temperature: float = 0.0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        raise NotImplementedError


class OpenAICompatible(ChatModel):
    """OpenAI chat-completions format; works for OpenAI, Gemini, Groq, Ollama, vLLM..."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 2,
    ):
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries

    def chat(self, messages, tools=None):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        t0 = time.time()
        retries = 0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    timeout=120,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                calls = [
                    {
                        "id": c["id"],
                        "name": c["function"]["name"],
                        "arguments": json.loads(c["function"]["arguments"] or "{}"),
                    }
                    for c in msg.get("tool_calls") or []
                ]
                usage = data.get("usage", {})
                return LLMReply(
                    msg.get("content") or "",
                    calls,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    time.time() - t0,
                    retries=retries,
                )
            except Exception as exc:
                last_exc = exc
                retries += 1
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_exc}") from last_exc


class Anthropic(ChatModel):
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_retries: int = 2,
    ):
        self.name = model
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries

    @staticmethod
    def _convert(messages):
        """OpenAI-style history -> Anthropic system + messages."""
        system, out = "", []
        for m in messages:
            if m["role"] == "system":
                system += m["content"]
            elif m["role"] == "tool":
                out.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                    ],
                })
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [
                    {
                        "type": "tool_use",
                        "id": c["id"],
                        "name": c["function"]["name"],
                        "input": json.loads(c["function"]["arguments"]),
                    }
                    for c in m["tool_calls"]
                ]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return system, out

    def chat(self, messages, tools=None):
        system, msgs = self._convert(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "system": system,
            "messages": msgs,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
        t0 = time.time()
        retries = 0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    json=body,
                    timeout=120,
                    headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                )
                r.raise_for_status()
                data = r.json()
                text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
                calls = [
                    {"id": b["id"], "name": b["name"], "arguments": b["input"]}
                    for b in data["content"]
                    if b["type"] == "tool_use"
                ]
                u = data.get("usage", {})
                return LLMReply(
                    text,
                    calls,
                    u.get("input_tokens", 0),
                    u.get("output_tokens", 0),
                    time.time() - t0,
                    retries=retries,
                )
            except Exception as exc:
                last_exc = exc
                retries += 1
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Anthropic call failed after {retries} retries: {last_exc}") from last_exc


class Scripted(ChatModel):
    """Deterministic stand-in for an LLM, used for offline smoke tests only.

    Tool-calling mode follows a fixed investigation script:
      Step 1: detect_metric_anomalies(top_k=6) + get_call_graph()
      Step 2: get_log_summary + get_trace_summary for the top 2 suspects
      Step 3: get_log_summary + get_trace_summary for suspects 3..4 (if budget allows)
      Step 4+: emit structured JSON ranking combining metric, log, and trace observations.
    Never report its numbers as LLM results.
    """

    name = "scripted"

    def __init__(self, seed: int = 42, temperature: float = 0.0):
        self.seed = seed
        self.temperature = temperature

    def chat(self, messages, tools=None):
        t0 = time.time()
        prompt_chars = sum(len(json.dumps(m)) for m in messages)
        est_in_tokens = max(64, prompt_chars // 4)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if tools and not tool_msgs:
            return LLMReply(
                tool_calls=[
                    {"id": "c0", "name": "detect_metric_anomalies", "arguments": {"top_k": 6}},
                    {"id": "c1", "name": "get_call_graph", "arguments": {}},
                ],
                input_tokens=est_in_tokens,
                output_tokens=42,
                latency_s=time.time() - t0,
            )

        all_text = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        suspects, metric_map = _suspects_and_metrics_from(all_text)

        if tools and len(tool_msgs) <= 2 and suspects:
            calls = []
            for i, s in enumerate(suspects[:2]):
                calls += [
                    {"id": f"l_{i}", "name": "get_log_summary", "arguments": {"service": s}},
                    {"id": f"t_{i}", "name": "get_trace_summary", "arguments": {"service": s}},
                ]
            return LLMReply(
                tool_calls=calls,
                input_tokens=est_in_tokens,
                output_tokens=68,
                latency_s=time.time() - t0,
            )

        if tools and len(tool_msgs) <= 6 and len(suspects) > 2:
            calls = []
            for i, s in enumerate(suspects[2:4], start=2):
                calls += [
                    {"id": f"l_{i}", "name": "get_log_summary", "arguments": {"service": s}},
                    {"id": f"t_{i}", "name": "get_trace_summary", "arguments": {"service": s}},
                ]
            return LLMReply(
                tool_calls=calls,
                input_tokens=est_in_tokens,
                output_tokens=64,
                latency_s=time.time() - t0,
            )

        # Synthesize ranking from observed tool outputs
        ranking = _score_observed_suspects(suspects, metric_map, tool_msgs, seed=self.seed)
        answer = {
            "ranking": ranking,
            "summary": "scripted offline investigation answer",
        }
        out_str = json.dumps(answer)
        return LLMReply(
            text=out_str,
            input_tokens=est_in_tokens,
            output_tokens=max(48, len(out_str) // 4),
            latency_s=time.time() - t0,
        )


def _suspects_and_metrics_from(text: str) -> tuple[list[str], dict[str, tuple[str, float]]]:
    key = text.find('"top_anomalies"')
    if key < 0:
        return [], {}
    try:
        rows = json.JSONDecoder().raw_decode(text[text.find("[", key) :])[0]
    except Exception:
        return [], {}
    seen: list[str] = []
    metric_map: dict[str, tuple[str, float]] = {}
    for r in rows:
        svc = r.get("service")
        if not svc:
            continue
        if svc not in seen:
            seen.append(svc)
            metric_map[svc] = (str(r.get("metric", "unknown")), float(r.get("score", 0.0)))
    return seen, metric_map


def _score_observed_suspects(
    suspects: list[str],
    metric_map: dict[str, tuple[str, float]],
    tool_msgs: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    log_bonus: dict[str, float] = {}
    trace_bonus: dict[str, float] = {}
    for m in tool_msgs:
        try:
            obj = json.loads(m.get("content", "{}"))
        except Exception:
            continue
        svc = obj.get("service")
        if not svc:
            continue
        if "error_lines_per_min" in obj:
            err = float(obj.get("error_lines_per_min") or 0.0)
            norm = float(obj.get("normal_error_lines_per_min") or 0.0)
            new_t = len(obj.get("new_templates") or [])
            if err > norm or new_t > 0:
                log_bonus[svc] = 0.25 * min(new_t, 4) + (0.35 if err > norm + 1.0 else 0.0)
        if "after_p90_ms" in obj and "normal_p90_ms" in obj:
            ap = float(obj.get("after_p90_ms") or 0.0)
            np90 = float(obj.get("normal_p90_ms") or 0.0)
            if np90 > 0 and ap / np90 > 1.5:
                trace_bonus[svc] = 0.30

    scored = []
    for idx, s in enumerate(suspects[:5]):
        m_name, m_val = metric_map.get(s, ("unknown", 1.0))
        # Seed-dependent micro-jitter to model realistic stochastic decoding across seeds when seed != 42
        h = int(hashlib.md5(f"{s}:{seed}".encode()).hexdigest()[:8], 16)
        rng = random.Random(h)
        jitter = 0.0 if seed == 42 else rng.uniform(-0.08, 0.08)
        score = (1.0 / (idx + 1)) + log_bonus.get(s, 0.0) + trace_bonus.get(s, 0.0) + jitter
        ft = "unknown"
        if m_name == "cpu":
            ft = "cpu_contention"
        elif m_name in ("server_latency_p90", "client_latency_p90", "net_rx", "net_tx"):
            ft = "network_delay"
        if log_bonus.get(s, 0.0) >= 0.5 and m_name != "cpu":
            ft = "exception"
        scored.append(
            (
                score,
                {
                    "service": s,
                    "fault_type": ft,
                    "confidence": round(min(max(0.85 - 0.12 * idx, 0.2), 0.95), 2),
                    "evidence": [f"top_metric={m_name} ({m_val:.1f})"],
                },
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


def get_model(
    provider: str,
    model: str | None = None,
    temperature: float = 0.0,
    seed: int = 42,
) -> ChatModel:
    load_dotenv()
    if provider == "scripted":
        return Scripted(seed=seed, temperature=temperature)
    if provider == "openai":
        return OpenAICompatible(
            model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ.get("OPENAI_API_KEY", "none"),
            temperature=temperature,
            seed=seed,
        )
    if provider == "gemini":
        return OpenAICompatible(
            model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "https://generativelanguage.googleapis.com/v1beta/openai",
            os.environ["GEMINI_API_KEY"],
            temperature=temperature,
            seed=seed,
        )
    if provider == "anthropic":
        return Anthropic(
            model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            os.environ["ANTHROPIC_API_KEY"],
            temperature=temperature,
            seed=seed,
        )
    raise ValueError(f"unknown provider {provider}")
