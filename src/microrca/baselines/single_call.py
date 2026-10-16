"""B1: single-LLM-call RCA (non-agentic).

All telemetry summaries are pre-computed and packed into one prompt; the model
answers once with the Diagnosis JSON. No tool choice, no iteration, no verification.
"""
from __future__ import annotations

import json

from microrca.agent.parsing import parse_diagnosis
from microrca.contracts import Diagnosis, Usage
from microrca.data.store import CaseBundle
from microrca.llm.providers import ChatModel
from microrca.tools.telemetry import detect_metric_anomalies, get_call_graph, get_log_summary, get_trace_summary

SYSTEM = (
    "You are an SRE diagnosing a failure in a microservice system. Identify the root-cause service "
    "and fault type (one of cpu_contention, cpu_consumed, network_delay, exception, return). "
    'Answer with JSON only: {"ranking": [{"service": str, "fault_type": str, "confidence": float, '
    '"evidence": [str]}], "summary": str}. Rank up to 5 services, most likely first.'
)


def build_prompt(bundle: CaseBundle, top_k: int = 8) -> str:
    anomalies = detect_metric_anomalies(bundle, top_k=top_k)
    suspects = list(dict.fromkeys(r["service"] for r in anomalies["top_anomalies"]))[:5]
    parts = {
        "system": bundle.case.system,
        "services": bundle.case.services,
        "call_graph": get_call_graph(bundle)["edges"],
        **anomalies,
        "logs": [get_log_summary(bundle, s) for s in suspects],
        "traces": [get_trace_summary(bundle, s) for s in suspects],
    }
    return "Telemetry summary for the incident:\n" + json.dumps(parts)


def diagnose(bundle: CaseBundle, llm: ChatModel) -> tuple[Diagnosis, Usage]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(bundle)}]
    reply = llm.chat(messages)
    usage = Usage(llm_calls=1, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                  latency_s=reply.latency_s)
    return parse_diagnosis(reply.text, bundle.case.case_id, f"B1-single-call:{llm.name}"), usage
