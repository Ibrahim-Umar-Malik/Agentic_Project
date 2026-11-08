"""Gap 3: Leakage-Free Incident Memory.

Hypothesis:
  Recurring failure modes in microservice architectures share structural telemetry
  signatures (e.g. which service/fault pairs manifest as log template bursts vs.
  trace slowdowns vs. CPU saturation). Retrieving similar historical incidents from
  a Dev-only episodic memory bank provides few-shot diagnostic priors and refines
  ambiguous service/fault rankings on unseen incidents.

CRITICAL NON-NEGOTIABLE RULE (No Data Leakage):
  1. `IncidentMemory` only indexes ground-truth labels from the **Dev** split
     (`split="dev"`). Attempting to index any case belonging to the `test` split
     raises a `ValueError`.
  2. During inference/evaluation on any test case, `store()` strips all ground-truth
     labels (`bundle.blind()`) and never persists test labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from microrca.contracts import AgentState, Diagnosis, FaultType, RankedCause
from microrca.data.store import CaseBundle, DEFAULT_SPLITS_DIR, load_split_ids
from microrca.gaps.verifier import clean_log_err_score
from microrca.tools.telemetry import KEY_METRICS, metric_score, trace_anomaly_score


@dataclass
class MemoryEntry:
    case_id: str
    system: str
    root_service: str
    fault_type: str
    fingerprint: list[float]
    top_signals: dict[str, str]
    is_labeled_dev: bool = True


def extract_fingerprint(bundle: CaseBundle) -> tuple[list[float], dict[str, str]]:
    """Compute a label-free telemetry feature vector for similarity retrieval."""
    services = bundle.case.services
    if not services:
        return [0.0] * 8, {}

    cpu_scores = {s: metric_score(bundle, s, KEY_METRICS["cpu"]) for s in services}
    mem_scores = {s: metric_score(bundle, s, KEY_METRICS["memory"]) for s in services}
    srv_scores = {s: metric_score(bundle, s, KEY_METRICS["server_latency_p90"]) for s in services}
    cli_scores = {s: metric_score(bundle, s, KEY_METRICS["client_latency_p90"]) for s in services}
    net_scores = {
        s: max(
            metric_score(bundle, s, KEY_METRICS["net_rx"]),
            metric_score(bundle, s, KEY_METRICS["net_tx"]),
        )
        for s in services
    }
    log_scores = {s: clean_log_err_score(bundle, s)[0] for s in services}
    trc_scores = {s: trace_anomaly_score(bundle, s) for s in services}

    vec = [
        math.log1p(max(cpu_scores.values(), default=0.0)),
        math.log1p(max(mem_scores.values(), default=0.0)),
        math.log1p(max(srv_scores.values(), default=0.0)),
        math.log1p(max(cli_scores.values(), default=0.0)),
        math.log1p(max(net_scores.values(), default=0.0)),
        math.log1p(max(log_scores.values(), default=0.0)),
        math.log1p(max(trc_scores.values(), default=0.0)),
        float(len(services)) / 50.0,
    ]
    top_signals = {
        "top_cpu_service": max(cpu_scores, key=cpu_scores.get),
        "top_lat_service": max(srv_scores, key=srv_scores.get),
        "top_log_service": max(log_scores, key=log_scores.get),
        "top_trace_service": max(trc_scores, key=trc_scores.get),
        "top_net_service": max(net_scores, key=net_scores.get),
    }
    return vec, top_signals


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return dot / (na * nb)


@dataclass
class IncidentMemory:
    """Episodic incident memory strictly isolated from test-split labels."""

    entries: list[MemoryEntry] = field(default_factory=list)
    unlabeled_runtime_traces: list[dict[str, Any]] = field(default_factory=list)
    _forbidden_test_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self._forbidden_test_ids:
            try:
                self._forbidden_test_ids = load_split_ids("test", DEFAULT_SPLITS_DIR)
            except Exception:
                self._forbidden_test_ids = set()

    @classmethod
    def from_dev_cases(
        cls,
        dev_cases: list[CaseBundle],
        forbidden_test_ids: set[str] | None = None,
    ) -> IncidentMemory:
        mem = cls(
            _forbidden_test_ids=forbidden_test_ids
            if forbidden_test_ids is not None
            else load_split_ids("test", DEFAULT_SPLITS_DIR)
        )
        mem.index_dev_cases(dev_cases)
        return mem

    def index_dev_cases(self, dev_cases: list[CaseBundle]) -> None:
        """Populate memory bank from Dev cases only. Raises ValueError on any test case."""
        for bundle in dev_cases:
            cid = bundle.case.case_id
            if cid in self._forbidden_test_ids:
                raise ValueError(
                    f"DATA LEAKAGE VIOLATION: attempted to index test split case {cid!r} into IncidentMemory!"
                )
            if bundle.case.label is None:
                raise ValueError(f"Dev case {cid!r} is missing ground-truth label for memory indexing.")
            vec, top_sig = extract_fingerprint(bundle)
            self.entries.append(
                MemoryEntry(
                    case_id=cid,
                    system=bundle.case.system,
                    root_service=bundle.case.label.service,
                    fault_type=bundle.case.label.fault_type.value,
                    fingerprint=vec,
                    top_signals=top_sig,
                    is_labeled_dev=True,
                )
            )

    def assert_no_test_leakage(self, test_ids: set[str] | None = None) -> bool:
        """Verify that zero test case IDs or test labels exist anywhere in memory."""
        check_ids = test_ids if test_ids is not None else self._forbidden_test_ids
        for entry in self.entries:
            if entry.case_id in check_ids:
                raise AssertionError(f"Leakage detected: test case {entry.case_id} found in memory entries!")
        for rt in self.unlabeled_runtime_traces:
            if "true_label" in rt or "label" in rt:
                raise AssertionError("Leakage detected: label field present in runtime memory trace!")
        return True

    def retrieve_entries(self, bundle: CaseBundle, k: int = 3) -> list[tuple[float, MemoryEntry]]:
        vec, top_sig = extract_fingerprint(bundle)
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self.entries:
            # Never retrieve the exact same case_id (leave-one-out when running on dev)
            if entry.case_id == bundle.case.case_id:
                continue
            if entry.system != bundle.case.system:
                continue
            sim = _cosine(vec, entry.fingerprint)
            # Bonus when the same service leads in log/trace/lat signals
            overlap = sum(
                1
                for key in ("top_cpu_service", "top_lat_service", "top_log_service", "top_trace_service")
                if entry.top_signals.get(key) == top_sig.get(key)
            )
            total_sim = sim + 0.06 * overlap
            scored.append((total_sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def recall(
        self, bundle: CaseBundle, k: int = 3, state: AgentState | None = None
    ) -> list[str]:
        matches = self.retrieve_entries(bundle, k=k)
        lines = []
        for sim, e in matches:
            lines.append(
                f"- [dev case {e.case_id}, sim={sim:.2f}] system={e.system}, "
                f"resolved_root_cause={e.root_service}, fault_type={e.fault_type}, "
                f"signals={e.top_signals}"
            )
        if state is not None:
            state.gap_events.append(
                {
                    "gap": "memory",
                    "step": state.step,
                    "action": "recall",
                    "retrieved_dev_cases": [e.case_id for _, e in matches],
                }
            )
        return lines

    def rerank_with_memory(
        self, bundle: CaseBundle, diagnosis: Diagnosis, state: AgentState | None = None
    ) -> Diagnosis:
        """Apply a soft prior from similar Dev incidents to break close ties and refine fault types."""
        matches = self.retrieve_entries(bundle, k=3)
        if not matches or not diagnosis.ranking:
            return diagnosis

        _, cur_signals = extract_fingerprint(bundle)
        service_bonus: dict[str, float] = {}
        fault_votes: dict[str, float] = {}
        for sim, entry in matches:
            if sim < 0.65:
                continue
            # If the historical root cause matched one of the current leading modality services, boost that service
            for sig_key, svc in cur_signals.items():
                if entry.top_signals.get(sig_key) == entry.root_service:
                    service_bonus[svc] = service_bonus.get(svc, 0.0) + 0.08 * sim
            fault_votes[entry.fault_type] = fault_votes.get(entry.fault_type, 0.0) + sim

        # Re-score the current ranking with small memory tie-break weight
        scored_causes: list[tuple[float, RankedCause]] = []
        for idx, cause in enumerate(diagnosis.ranking):
            base = 1.0 / (idx + 1) + 0.35 * cause.confidence
            bonus = service_bonus.get(cause.service, 0.0)
            ev = list(cause.evidence)
            if bonus > 0:
                ev.append(f"memory_prior_bonus=+{bonus:.2f}")
            ft = cause.fault_type
            if ft == FaultType.UNKNOWN and fault_votes:
                best_ft = max(fault_votes, key=fault_votes.get)
                if best_ft in FaultType._value2member_map_:
                    ft = FaultType(best_ft)
            scored_causes.append(
                (
                    base + bonus,
                    RankedCause(
                        service=cause.service,
                        fault_type=ft,
                        confidence=min(0.99, round(cause.confidence + 0.5 * bonus, 2)),
                        evidence=ev,
                    ),
                )
            )
        scored_causes.sort(key=lambda x: x[0], reverse=True)
        new_ranking = [c for _, c in scored_causes]

        if state is not None:
            state.gap_events.append(
                {
                    "gap": "memory",
                    "step": state.step,
                    "action": "rerank",
                    "top1_before": diagnosis.ranking[0].service,
                    "top1_after": new_ranking[0].service,
                }
            )
        method_name = diagnosis.method if "+memory" in diagnosis.method else f"{diagnosis.method}+memory"
        return diagnosis.model_copy(update={"method": method_name, "ranking": new_ranking})

    def store(
        self, bundle: CaseBundle, diagnosis: Diagnosis, state: AgentState | None = None
    ) -> None:
        """Record an unlabeled runtime trace (NEVER stores ground-truth labels)."""
        blind_case = bundle.blind()
        assert blind_case.label is None, "blind() must strip label before storing runtime trace"
        self.unlabeled_runtime_traces.append(
            {
                "case_id": blind_case.case_id,
                "system": blind_case.system,
                "predicted_top1": diagnosis.ranking[0].service if diagnosis.ranking else None,
                "predicted_fault": diagnosis.ranking[0].fault_type.value if diagnosis.ranking else "unknown",
            }
        )
        if state is not None:
            state.gap_events.append(
                {
                    "gap": "memory",
                    "step": state.step,
                    "action": "store_unlabeled",
                    "case_id": blind_case.case_id,
                }
            )
