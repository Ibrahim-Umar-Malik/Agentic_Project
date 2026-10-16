"""Proposed system, Phase 1 skeleton: a tool-using ReAct-style RCA agent.

Loop: the model sees the incident, chooses observability tools, reads their results,
and stops when it emits a Diagnosis JSON (or hits the step / tool budget).

Hooks for the three research gaps (filled in Phase 2):
  * Gap 1, verification: `verifier` re-checks the top hypothesis against telemetry
    before the answer is accepted (counter-evidence check / self-consistency).
  * Gap 2, cost-aware investigation: `max_steps` and `max_tool_calls` budgets, with
    per-run token and latency accounting in AgentState.usage.
  * Gap 3, incident memory: `memory` retrieves similar past incidents and injects
    them as context; new diagnoses are written back.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Protocol

from microrca.agent.parsing import parse_diagnosis
from microrca.contracts import AgentState, Diagnosis, ToolResult
from microrca.data.store import CaseBundle
from microrca.llm.providers import ChatModel
from microrca.tools.telemetry import TOOL_SPECS, call_tool

SYSTEM = (
    "You are an SRE agent diagnosing a failure in a microservice system. Use the tools to inspect "
    "metrics, logs, traces and the call graph. Investigate before answering: check the strongest "
    "anomalies and whether they are a cause or a downstream symptom. Fault types: cpu_contention, "
    "cpu_consumed, network_delay, exception, return. When confident, reply with JSON only: "
    '{"ranking": [{"service": str, "fault_type": str, "confidence": float, "evidence": [str]}], '
    '"summary": str} ranking up to 5 services.'
)


class Memory(Protocol):
    def recall(self, bundle: CaseBundle, k: int = 3) -> list[str]: ...
    def store(self, bundle: CaseBundle, diagnosis: Diagnosis) -> None: ...


Verifier = Callable[[CaseBundle, Diagnosis], Diagnosis]


def run(bundle: CaseBundle, llm: ChatModel, max_steps: int = 8, max_tool_calls: int = 20,
        memory: Memory | None = None, verifier: Verifier | None = None) -> tuple[Diagnosis, AgentState]:
    case = bundle.blind()
    state = AgentState(case=case, max_steps=max_steps)
    intro = f"Incident in {case.system} at unix time {case.alert_time}. Services: {', '.join(case.services)}."
    if memory:
        past = memory.recall(bundle)
        if past:
            intro += "\nSimilar past incidents:\n" + "\n".join(past)
    state.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": intro}]

    final_text = ""
    t0 = time.time()
    while not state.done and state.step < state.max_steps:
        state.step += 1
        budget_left = state.usage.tool_calls < max_tool_calls
        reply = llm.chat(state.messages, tools=TOOL_SPECS if budget_left else None)
        state.usage.llm_calls += 1
        state.usage.input_tokens += reply.input_tokens
        state.usage.output_tokens += reply.output_tokens
        if not reply.tool_calls:
            final_text, state.done = reply.text, True
            break
        state.messages.append({"role": "assistant", "content": reply.text, "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in reply.tool_calls]})
        for c in reply.tool_calls:
            result = call_tool(bundle, c["name"], c["arguments"])
            state.usage.tool_calls += 1
            state.observations.append(ToolResult(name=c["name"], ok="error" not in result, content=result))
            state.messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)[:4000]})

    if not state.done:  # budget exhausted: force an answer without tools
        state.messages.append({"role": "user", "content": "Budget exhausted. Give your final JSON answer now."})
        reply = llm.chat(state.messages)
        state.usage.llm_calls += 1
        state.usage.input_tokens += reply.input_tokens
        state.usage.output_tokens += reply.output_tokens
        final_text = reply.text
    state.usage.latency_s = time.time() - t0

    diagnosis = parse_diagnosis(final_text, case.case_id, f"P0-react-agent:{llm.name}")
    if verifier:
        diagnosis = verifier(bundle, diagnosis)
    if memory:
        memory.store(bundle, diagnosis)
    state.hypotheses = diagnosis.ranking
    return diagnosis, state
