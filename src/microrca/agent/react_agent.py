"""Proposed system: a tool-using ReAct-style RCA agent with toggleable Phase 2 Gap modules.

Loop: the model sees the incident (strictly blinded via `bundle.blind_bundle()`), chooses
observability tools, reads their results, and stops when it emits a Diagnosis JSON
(or when the adaptive budget controller / step cap converges).

Toggleable research-gap modules:
  * Gap 1 (Verification, `gaps/verifier.py`): `verifier` re-checks the candidate ranking
    against multi-modal telemetry and call-graph propagation before accepting the answer.
  * Gap 2 (Cost-aware investigation, `gaps/budget.py`): `budget_controller` caches duplicate
    tool calls, limits fan-out, and triggers early stopping on multi-modal convergence.
  * Gap 3 (Incident memory, `gaps/memory.py`): `memory` retrieves similar Dev-only incidents
    as few-shot context and applies a prior tie-breaker without test-set leakage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from microrca.agent.parsing import parse_diagnosis
from microrca.contracts import AgentState, Diagnosis, ToolResult
from microrca.data.store import CaseBundle
from microrca.gaps.budget import BudgetConfig, BudgetController
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


Verifier = Callable[..., Diagnosis]


def write_trajectory_log(
    log_dir: Path,
    run_id: str,
    state: AgentState,
    diagnosis: Diagnosis,
    config_meta: dict[str, Any] | None = None,
) -> Path:
    """Write full step-by-step agent trajectory to `logs/<run_id>/<case_id>.jsonl`."""
    run_folder = Path(log_dir) / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    if config_meta:
        cfg_file = run_folder / "config.json"
        if not cfg_file.exists():
            cfg_file.write_text(json.dumps(config_meta, indent=2))

    out_path = run_folder / f"{state.case.case_id}.jsonl"
    events: list[dict[str, Any]] = [
        {
            "event": "run_start",
            "run_id": run_id,
            "case_id": state.case.case_id,
            "system": state.case.system,
            "seed": state.seed,
            "max_steps": state.max_steps,
            "config": config_meta or {},
        }
    ]
    for idx, msg in enumerate(state.messages):
        events.append({"event": "message", "index": idx, **msg})
    for idx, obs in enumerate(state.observations):
        events.append({"event": "observation", "index": idx, **obs.model_dump()})
    for ge in state.gap_events:
        events.append({"event": "gap_action", **ge})
    events.append(
        {
            "event": "run_end",
            "case_id": state.case.case_id,
            "method": diagnosis.method,
            "verified": diagnosis.verified,
            "verification_notes": diagnosis.verification_notes,
            "ranking": [c.model_dump() for c in diagnosis.ranking],
            "usage": state.usage.model_dump(),
            "error": state.error,
        }
    )
    with out_path.open("w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return out_path


def run(
    bundle: CaseBundle,
    llm: ChatModel,
    max_steps: int = 8,
    max_tool_calls: int = 20,
    memory: Any | None = None,
    verifier: Verifier | None = None,
    budget_controller: BudgetController | None = None,
    seed: int = 42,
    run_id: str = "",
    log_dir: Path | None = None,
    method_tag: str | None = None,
) -> tuple[Diagnosis, AgentState]:
    # Strictly blind the bundle so ground-truth labels cannot leak into any tool or gap module
    blind = bundle.blind_bundle()
    case = blind.case
    assert case.label is None, "Agent must operate on blinded IncidentCase (label=None)"

    effective_max_steps = budget_controller.config.max_steps if budget_controller else max_steps
    effective_max_tools = budget_controller.config.max_tool_calls if budget_controller else max_tool_calls

    state = AgentState(case=case, max_steps=effective_max_steps, seed=seed, run_id=run_id)
    intro = f"Incident in {case.system} at unix time {case.alert_time}. Services: {', '.join(case.services)}."
    if memory is not None:
        try:
            past = memory.recall(blind, state=state)
        except TypeError:
            past = memory.recall(blind)
        if past:
            intro += "\nSimilar past incidents (from dev memory bank):\n" + "\n".join(past)
    state.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": intro}]

    final_text = ""
    t0 = time.time()
    while not state.done and state.step < state.max_steps:
        state.step += 1
        if budget_controller is not None:
            budget_left = budget_controller.can_call_tools(state)
        else:
            budget_left = state.usage.tool_calls < effective_max_tools

        try:
            reply = llm.chat(state.messages, tools=TOOL_SPECS if budget_left else None)
            state.usage.llm_calls += 1
            state.usage.input_tokens += reply.input_tokens
            state.usage.output_tokens += reply.output_tokens
            state.usage.retries += getattr(reply, "retries", 0)
        except Exception as exc:
            state.error = f"step {state.step} llm error: {exc}"
            state.usage.retries += 1
            break

        if not reply.tool_calls:
            final_text, state.done = reply.text, True
            break

        calls_to_run = (
            budget_controller.filter_tool_calls(reply.tool_calls, state)
            if budget_controller is not None
            else reply.tool_calls[: max(0, effective_max_tools - state.usage.tool_calls)]
        )
        if not calls_to_run:
            # Tool budget reached; move to final synthesis
            break

        state.messages.append(
            {
                "role": "assistant",
                "content": reply.text,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                    }
                    for c in calls_to_run
                ],
            }
        )
        for c in calls_to_run:
            if budget_controller is not None:
                result, from_cache = budget_controller.execute_or_cache(
                    call_tool, blind, c["name"], c["arguments"], state
                )
                if not from_cache:
                    state.usage.tool_calls += 1
            else:
                result = call_tool(blind, c["name"], c["arguments"])
                state.usage.tool_calls += 1
            state.observations.append(
                ToolResult(name=c["name"], ok="error" not in result, content=result)
            )
            state.messages.append(
                {"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)[:4000]}
            )

        if budget_controller is not None and budget_controller.should_stop_early(state):
            state.messages.append(
                {
                    "role": "user",
                    "content": "Multi-modal evidence converged across top suspects. Give your final JSON answer now.",
                }
            )
            try:
                reply = llm.chat(state.messages, tools=None)
                state.usage.llm_calls += 1
                state.usage.input_tokens += reply.input_tokens
                state.usage.output_tokens += reply.output_tokens
                state.usage.retries += getattr(reply, "retries", 0)
                final_text = reply.text
                state.done = True
            except Exception as exc:
                state.error = f"early-stop synthesis error: {exc}"
            break

    if not state.done:  # budget exhausted or error: force an answer without tools
        state.messages.append(
            {"role": "user", "content": "Budget exhausted. Give your final JSON answer now."}
        )
        try:
            reply = llm.chat(state.messages, tools=None)
            state.usage.llm_calls += 1
            state.usage.input_tokens += reply.input_tokens
            state.usage.output_tokens += reply.output_tokens
            state.usage.retries += getattr(reply, "retries", 0)
            final_text = reply.text
            state.done = True
        except Exception as exc:
            state.error = f"final synthesis error: {exc}"

    state.usage.latency_s = time.time() - t0
    if budget_controller is not None:
        budget_controller.finalize_usage(state)
    else:
        BudgetController().finalize_usage(state)

    base_method = method_tag or f"P0-react-agent:{llm.name}"
    diagnosis = parse_diagnosis(final_text, case.case_id, base_method)
    if not diagnosis.ranking and state.error is None:
        state.error = "unparseable_or_empty_diagnosis"

    if verifier is not None:
        try:
            diagnosis = verifier(blind, diagnosis, state=state)
        except TypeError:
            diagnosis = verifier(blind, diagnosis)
    if memory is not None:
        if hasattr(memory, "rerank_with_memory"):
            diagnosis = memory.rerank_with_memory(blind, diagnosis, state=state)
        try:
            memory.store(blind, diagnosis, state=state)
        except TypeError:
            memory.store(blind, diagnosis)

    if method_tag:
        diagnosis = diagnosis.model_copy(update={"method": method_tag})
    state.hypotheses = diagnosis.ranking

    if log_dir is not None and run_id:
        write_trajectory_log(
            log_dir=log_dir,
            run_id=run_id,
            state=state,
            diagnosis=diagnosis,
            config_meta={
                "method": diagnosis.method,
                "llm": llm.name,
                "seed": seed,
                "max_steps": state.max_steps,
                "max_tool_calls": effective_max_tools,
                "verifier": verifier is not None,
                "budget": budget_controller is not None,
                "memory": memory is not None,
            },
        )
    return diagnosis, state
