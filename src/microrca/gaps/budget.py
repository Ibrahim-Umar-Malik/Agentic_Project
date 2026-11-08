"""Gap 2: Adaptive Cost-Aware Budget Controller.

Hypothesis:
  Unbounded or fixed-step ReAct loops waste tokens and tool calls by re-querying
  already-inspected services or continuing exploration after a high-confidence
  multi-modal consensus has already formed.
  An adaptive budget controller that (1) caches & deduplicates identical tool calls,
  (2) dynamically caps per-step tool fan-out, and (3) triggers early convergence
  once top suspects are corroborated across metrics, logs, and traces reduces
  token/tool cost and latency without sacrificing AC@1 accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from microrca.contracts import AgentState


@dataclass
class BudgetConfig:
    """Adaptive budget policy parameters calibrated on the Dev split."""

    max_steps: int = 6
    max_tool_calls: int = 10
    max_calls_per_step: int = 4
    dominant_z_threshold: float = 25.0
    min_suspects_covered: int = 2
    enable_cache: bool = True
    enable_early_stop: bool = True
    # Standard per-1M-token pricing estimate (gpt-4o-mini / flash class default)
    input_cost_per_1m: float = 0.15
    output_cost_per_1m: float = 0.60


@dataclass
class BudgetController:
    config: BudgetConfig = field(default_factory=BudgetConfig)
    _cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _inspected_logs: set[str] = field(default_factory=set)
    _inspected_traces: set[str] = field(default_factory=set)
    _top_suspects: list[str] = field(default_factory=list)
    _max_anomaly_z: float = 0.0
    cache_hits: int = 0

    @staticmethod
    def _cache_key(name: str, arguments: dict[str, Any]) -> str:
        return f"{name}:{json.dumps(arguments, sort_keys=True)}"

    def can_call_tools(self, state: AgentState) -> bool:
        return state.usage.tool_calls < self.config.max_tool_calls and not self.should_stop_early(state)

    def filter_tool_calls(
        self, calls: list[dict[str, Any]], state: AgentState
    ) -> list[dict[str, Any]]:
        """Cap tool fan-out per step and respect remaining tool budget."""
        remaining = max(0, self.config.max_tool_calls - state.usage.tool_calls)
        limit = min(remaining, self.config.max_calls_per_step)
        trimmed = calls[:limit]
        if len(trimmed) < len(calls):
            state.gap_events.append(
                {
                    "gap": "budget",
                    "step": state.step,
                    "action": "trim_fanout",
                    "requested": len(calls),
                    "allowed": len(trimmed),
                }
            )
        return trimmed

    def execute_or_cache(
        self,
        call_fn,
        bundle,
        name: str,
        arguments: dict[str, Any],
        state: AgentState,
    ) -> tuple[dict[str, Any], bool]:
        """Run tool or return cached observation if identical call was already made."""
        key = self._cache_key(name, arguments)
        if self.config.enable_cache and key in self._cache:
            self.cache_hits += 1
            state.gap_events.append(
                {
                    "gap": "budget",
                    "step": state.step,
                    "action": "cache_hit",
                    "tool": name,
                    "arguments": arguments,
                }
            )
            return self._cache[key], True

        result = call_fn(bundle, name, arguments)
        self._cache[key] = result
        self._record_observation(name, arguments, result)
        return result, False

    def _record_observation(
        self, name: str, arguments: dict[str, Any], result: dict[str, Any]
    ) -> None:
        if name == "detect_metric_anomalies" and "top_anomalies" in result:
            rows = result["top_anomalies"]
            if rows:
                self._max_anomaly_z = max(float(r.get("score", 0.0)) for r in rows)
                seen: list[str] = []
                for r in rows:
                    svc = r.get("service")
                    if svc and svc not in seen:
                        seen.append(svc)
                self._top_suspects = seen
        elif name == "get_log_summary" and "service" in arguments:
            self._inspected_logs.add(str(arguments["service"]))
        elif name == "get_trace_summary" and "service" in arguments:
            self._inspected_traces.add(str(arguments["service"]))

    def should_stop_early(self, state: AgentState) -> bool:
        """Stop tool exploration early when top suspects have multi-modal coverage."""
        if not self.config.enable_early_stop:
            return False
        if not self._top_suspects:
            return False
        # Fast path: overwhelming metric anomaly + at least 1 suspect log/trace check
        top1 = self._top_suspects[0]
        if (
            self._max_anomaly_z >= self.config.dominant_z_threshold
            and top1 in self._inspected_logs
            and top1 in self._inspected_traces
        ):
            state.usage.early_stop = True
            state.gap_events.append(
                {
                    "gap": "budget",
                    "step": state.step,
                    "action": "early_stop_dominant",
                    "top_suspect": top1,
                    "max_z": round(self._max_anomaly_z, 2),
                }
            )
            return True

        # Standard convergence: top K suspects inspected across both logs and traces
        target = self._top_suspects[: self.config.min_suspects_covered]
        if target and all(s in self._inspected_logs and s in self._inspected_traces for s in target):
            state.usage.early_stop = True
            state.gap_events.append(
                {
                    "gap": "budget",
                    "step": state.step,
                    "action": "early_stop_converged",
                    "covered_suspects": target,
                }
            )
            return True
        return False

    def finalize_usage(self, state: AgentState) -> None:
        u = state.usage
        cost = (
            (u.input_tokens / 1_000_000.0) * self.config.input_cost_per_1m
            + (u.output_tokens / 1_000_000.0) * self.config.output_cost_per_1m
        )
        u.estimated_cost_usd = round(cost, 6)
