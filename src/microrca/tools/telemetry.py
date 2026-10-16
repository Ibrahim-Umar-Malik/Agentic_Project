"""Observability tools the agent can call.

Each tool reads only the telemetry of the current incident (no labels) and returns
small, JSON-serialisable results so they fit in an LLM context window. The same
functions back the statistical baseline, the single-call LLM baseline (which gets
their output pre-computed) and the tool-using agent (which calls them on demand).
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from microrca.data.store import CaseBundle

KEY_METRICS = {
    "cpu": "CpuUsageRate(%)",
    "memory": "MemoryUsageRate(%)",
    "net_rx": "NetworkReceiveBytes",
    "net_tx": "NetworkTransmitBytes",
    "client_latency_p90": "PodClientLatencyP90(s)",
    "server_latency_p90": "PodServerLatencyP90(s)",
    "workload": "PodWorkload(Ops)",
    "success_rate": "PodSuccessRate(%)",
}


def _split(bundle: CaseBundle, service: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    m = bundle.metrics[service]
    t = bundle.case.alert_time
    minutes = np.array(m["minutes"])
    vals = np.array([np.nan if v is None else v for v in m[column]], dtype=float)
    alert_min = (t // 60) * 60
    return vals[minutes < alert_min], vals[minutes >= alert_min]


def metric_score(bundle: CaseBundle, service: str, column: str) -> float:
    """Robust z-score of the post-alert peak deviation versus the pre-alert window."""
    pre, post = _split(bundle, service, column)
    pre, post = pre[~np.isnan(pre)], post[~np.isnan(post)]
    if len(pre) < 2 or len(post) == 0:
        return 0.0
    med = float(np.median(pre))
    iqr = float(np.percentile(pre, 75) - np.percentile(pre, 25))
    # Floor the scale so flat or sparse series (e.g. idle latency gauges) cannot explode the score.
    scale = max(iqr / 1.349, float(np.std(pre)), 0.05 * abs(med), 1e-3)
    return min(float(np.max(np.abs(post - med)) / scale), 1000.0)


# ----------------------------------------------------------------------------- tools


def list_services(bundle: CaseBundle) -> dict[str, Any]:
    return {"system": bundle.case.system, "services": bundle.case.services}


def get_call_graph(bundle: CaseBundle, service: str | None = None) -> dict[str, Any]:
    edges = bundle.case.call_edges
    if service:
        return {
            "service": service,
            "calls": sorted({b for a, b in edges if a == service}),
            "called_by": sorted({a for a, b in edges if b == service}),
        }
    return {"edges": [list(e) for e in edges]}


def detect_metric_anomalies(bundle: CaseBundle, top_k: int = 10) -> dict[str, Any]:
    """Rank (service, metric) pairs by how far they moved after the alert."""
    rows = []
    for svc in bundle.case.services:
        for name, col in KEY_METRICS.items():
            s = metric_score(bundle, svc, col)
            if math.isfinite(s):
                rows.append({"service": svc, "metric": name, "score": round(s, 2)})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"top_anomalies": rows[:top_k]}


def get_metric_summary(bundle: CaseBundle, service: str, metric: str = "cpu") -> dict[str, Any]:
    col = KEY_METRICS.get(metric, metric)
    if service not in bundle.metrics or col not in bundle.metrics[service]:
        return {"error": f"unknown service/metric {service}/{metric}"}
    pre, post = _split(bundle, service, col)
    clean = lambda a: [round(float(x), 4) for x in a if not np.isnan(x)]  # noqa: E731
    pre_c = clean(pre)
    return {
        "service": service,
        "metric": metric,
        "before_mean": round(float(np.mean(pre_c)), 4) if pre_c else None,
        "before_std": round(float(np.std(pre_c)), 4) if pre_c else None,
        "after_values": clean(post),
        "anomaly_score": round(metric_score(bundle, service, col), 2),
    }


def get_log_summary(bundle: CaseBundle, service: str, top_k: int = 5) -> dict[str, Any]:
    fault = bundle.logs["fault"].get(service)
    normal = bundle.logs["normal"].get(service)
    if not fault:
        return {"service": service, "note": "no logs emitted after the alert"}
    normal_tpl = set((normal or {}).get("templates", {}))
    new = [t for t in fault["templates"] if t not in normal_tpl][:top_k]
    return {
        "service": service,
        "lines_per_min": round(fault["lines_per_min"], 1),
        "normal_lines_per_min": round(normal["lines_per_min"], 1) if normal else None,
        "error_lines_per_min": round(fault["error_lines_per_min"], 1),
        "normal_error_lines_per_min": round(normal["error_lines_per_min"], 1) if normal else None,
        "new_templates": new,
    }


def get_trace_summary(bundle: CaseBundle, service: str) -> dict[str, Any]:
    fault = bundle.traces["fault"].get(service)
    normal = bundle.traces["normal"].get(service)
    if not fault:
        return {"service": service, "note": "no spans after the alert"}
    out = {"service": service} | {f"after_{k}": round(v, 3) for k, v in fault.items()}
    if normal:
        out |= {f"normal_{k}": round(v, 3) for k, v in normal.items()}
    return out


def log_anomaly_score(bundle: CaseBundle, service: str) -> float:
    fault = bundle.logs["fault"].get(service)
    normal = bundle.logs["normal"].get(service)
    if not fault:
        return 0.0
    base_err = normal["error_lines_per_min"] if normal else 0.0
    err = math.log1p(max(fault["error_lines_per_min"] - base_err, 0.0))
    new = len([t for t in fault["templates"] if t not in set((normal or {}).get("templates", {}))])
    return err + 0.3 * new


def trace_anomaly_score(bundle: CaseBundle, service: str) -> float:
    fault = bundle.traces["fault"].get(service)
    normal = bundle.traces["normal"].get(service)
    if not fault or not normal or normal["p90_ms"] <= 0:
        return 0.0
    return max(math.log(fault["p90_ms"] / normal["p90_ms"]), 0.0)


Tool = Callable[..., dict[str, Any]]

TOOLS: dict[str, Tool] = {
    "list_services": list_services,
    "get_call_graph": get_call_graph,
    "detect_metric_anomalies": detect_metric_anomalies,
    "get_metric_summary": get_metric_summary,
    "get_log_summary": get_log_summary,
    "get_trace_summary": get_trace_summary,
}

# JSON-schema tool specs (OpenAI / Anthropic function-calling format).
TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "list_services", "description": "List all services of the system under diagnosis.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_call_graph", "description": "Service call graph. With `service`, return its callers and callees.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}}},
    {"name": "detect_metric_anomalies",
     "description": "Rank (service, metric) pairs by robust z-score of change after the alert.",
     "parameters": {"type": "object", "properties": {"top_k": {"type": "integer", "default": 10}}}},
    {"name": "get_metric_summary",
     "description": f"Before/after values of one metric of one service. metric in {sorted(KEY_METRICS)}.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}, "metric": {"type": "string"}},
                    "required": ["service", "metric"]}},
    {"name": "get_log_summary",
     "description": "Log volume, error-like lines and log templates not seen in fault-free logs for a service.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}},
    {"name": "get_trace_summary",
     "description": "Span rate and p50/p90 span latency for a service, after the alert and in fault-free traffic.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}},
]


def call_tool(bundle: CaseBundle, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        return TOOLS[name](bundle, **arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
