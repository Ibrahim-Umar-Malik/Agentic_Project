"""RCA evaluation metrics (as used in RCAEval / Nezha / OpenRCA style studies) and multi-seed aggregation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from microrca.contracts import EvalRecord


def ac_at_k(records: list[EvalRecord], k: int) -> float:
    """AC@k: fraction of cases whose true root-cause service is in the top-k."""
    if not records:
        return 0.0
    return sum(r.true_service in r.predicted[:k] for r in records) / len(records)


def avg_at_k(records: list[EvalRecord], k: int = 5) -> float:
    """Avg@k: mean of AC@1..AC@k."""
    return sum(ac_at_k(records, j) for j in range(1, k + 1)) / k


def fault_type_accuracy(records: list[EvalRecord]) -> float:
    hits = [r for r in records if r.predicted and r.predicted[0] == r.true_service]
    if not hits:
        return 0.0
    return sum(r.predicted_fault == r.true_fault for r in hits) / len(hits)


def summarize(records: list[EvalRecord]) -> dict[str, float]:
    n = len(records) or 1
    return {
        "cases": len(records),
        "AC@1": round(ac_at_k(records, 1), 3),
        "AC@3": round(ac_at_k(records, 3), 3),
        "AC@5": round(ac_at_k(records, 5), 3),
        "Avg@5": round(avg_at_k(records, 5), 3),
        "fault_type_acc|top1_hit": round(fault_type_accuracy(records), 3),
        "avg_llm_calls": round(sum(r.usage.llm_calls for r in records) / n, 2),
        "avg_tool_calls": round(sum(r.usage.tool_calls for r in records) / n, 2),
        "avg_tokens": round(sum(r.usage.input_tokens + r.usage.output_tokens for r in records) / n, 1),
        "avg_latency_s": round(sum(r.usage.latency_s for r in records) / n, 3),
        "avg_retries": round(sum(r.usage.retries for r in records) / n, 3),
        "avg_cost_usd": round(sum(r.usage.estimated_cost_usd for r in records) / n, 6),
        "early_stop_rate": round(sum(1 for r in records if r.usage.early_stop) / n, 3),
    }


def by_group(records: list[EvalRecord], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in records:
        groups[getattr(r, key)].append(r)
    return {g: summarize(rs) for g, rs in sorted(groups.items())}


def aggregate_seed_summaries(seed_summaries: list[dict[str, float]]) -> dict[str, Any]:
    """Compute mean, std, and 'mean ± std' formatted strings across multiple seeds."""
    if not seed_summaries:
        return {"mean": {}, "std": {}, "formatted": {}}
    keys = [k for k in seed_summaries[0] if k != "cases"]
    mean_dict: dict[str, float] = {"cases": seed_summaries[0].get("cases", 0)}
    std_dict: dict[str, float] = {}
    fmt_dict: dict[str, str] = {}
    for k in keys:
        vals = [float(s[k]) for s in seed_summaries if k in s]
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        decimals = 6 if "cost" in k else (1 if k == "avg_tokens" else 3)
        mean_dict[k] = round(m, decimals)
        std_dict[k] = round(sd, decimals)
        fmt_dict[k] = f"{m:.{decimals}f} ± {sd:.{decimals}f}"
    return {"mean": mean_dict, "std": std_dict, "formatted": fmt_dict}
