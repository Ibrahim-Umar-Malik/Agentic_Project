"""B0: non-agentic statistical RCA.

B0-stat-metric scores every service by its largest robust z-score across key metrics
(the idea behind BARO / N-Sigma detectors used as baselines in RCAEval).
B0-stat-multimodal ranks services separately on CPU, memory, server latency, trace
latency and log deviation, then fuses the rankings with reciprocal-rank fusion.
No LLM, no tools, no reasoning loop.
"""
from __future__ import annotations

from microrca.contracts import Diagnosis, FaultType, RankedCause
from microrca.data.store import CaseBundle
from microrca.tools.telemetry import KEY_METRICS, log_anomaly_score, metric_score, trace_anomaly_score

METRIC_TO_FAULT = {
    "cpu": FaultType.CPU_CONTENTION,
    "client_latency_p90": FaultType.NETWORK_DELAY,
    "server_latency_p90": FaultType.NETWORK_DELAY,
    "net_rx": FaultType.NETWORK_DELAY,
    "net_tx": FaultType.NETWORK_DELAY,
    "trace": FaultType.NETWORK_DELAY,
    "log": FaultType.EXCEPTION,
}
FUSED_SIGNALS = ["cpu", "memory", "server_latency_p90", "trace", "log"]


def _signal(bundle: CaseBundle, service: str, name: str) -> float:
    if name == "log":
        return log_anomaly_score(bundle, service)
    if name == "trace":
        return trace_anomaly_score(bundle, service)
    return metric_score(bundle, service, KEY_METRICS[name])


def _metric_only(bundle: CaseBundle) -> list[RankedCause]:
    causes = []
    for svc in bundle.case.services:
        scores = {name: metric_score(bundle, svc, col) for name, col in KEY_METRICS.items()}
        top = max(scores, key=scores.get)
        causes.append((scores[top], RankedCause(service=svc, fault_type=METRIC_TO_FAULT.get(top, FaultType.UNKNOWN),
                                                evidence=[f"{top} z={scores[top]:.1f}"])))
    causes.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in causes]


def _fused(bundle: CaseBundle) -> list[RankedCause]:
    services = bundle.case.services
    fused = {s: 0.0 for s in services}
    best: dict[str, tuple[int, str]] = {}
    for name in FUSED_SIGNALS:
        scores = {s: _signal(bundle, s, name) for s in services}
        for rank, s in enumerate(sorted(services, key=scores.get, reverse=True), start=1):
            fused[s] += 1.0 / rank
            if s not in best or rank < best[s][0]:
                best[s] = (rank, name)
    order = sorted(services, key=fused.get, reverse=True)
    return [RankedCause(service=s, fault_type=METRIC_TO_FAULT.get(best[s][1], FaultType.UNKNOWN),
                        evidence=[f"rrf={fused[s]:.2f}", f"strongest signal: {best[s][1]} (rank {best[s][0]})"])
            for s in order]


def diagnose(bundle: CaseBundle, multimodal: bool = False) -> Diagnosis:
    ranking = _fused(bundle) if multimodal else _metric_only(bundle)
    name = "B0-stat-multimodal" if multimodal else "B0-stat-metric"
    return Diagnosis(case_id=bundle.case.case_id, method=name, ranking=ranking)
