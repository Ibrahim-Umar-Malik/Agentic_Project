"""Gap 1: Multi-modal Counter-Evidence & Call-Graph Propagation Verifier.

Hypothesis:
  Single-pass and unverified ReAct agents frequently:
    (1) rank noisy chatty services whose `INFO` logs contain unmasked hex/UUID fragments
        above services with genuine error-rate surges (`error_lines_per_min`),
    (2) rank upstream callers (which log downstream RPC failures or suffer propagated
        client latency) above the true downstream callee root cause, and
    (3) misclassify fault types when CPU, latency, and log signals co-occur.
  A dedicated verification pass that filters out benign UUID template noise, checks
  multi-modal counter-evidence across CPU, server latency, network I/O, trace slowdown,
  and true error-log surges, and propagates blame along `caller -> callee` edges
  significantly improves both AC@1/Avg@5 and fault-type accuracy.

Note: All hyperparameters in `VerifierConfig` were calibrated strictly on the
30-case Nezha Dev split (`data/splits/dev.txt`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from microrca.contracts import AgentState, Diagnosis, FaultType, RankedCause
from microrca.data.store import CaseBundle
from microrca.tools.telemetry import (
    EXCEPTION_KEYWORDS,
    KEY_METRICS,
    RETURN_KEYWORDS,
    get_log_summary,
    metric_score,
    trace_anomaly_score,
    trace_volume_drop_score,
)

HEX_UUID_RE = re.compile(r"[0-9a-f]{4,}(?:-[0-9a-f<\*>]+)+|\b[0-9a-f]{6,}\b", re.I)
ERR_TPL_RE = re.compile(
    r"error|exception|warn|fail|refused|timeout|not found|invalid|null|panic|rpc error|unavail",
    re.I,
)

# Keywords in caller error templates that explicitly point to a specific downstream callee
CALLEE_HINTS = {
    "ads": "adservice",
    "adservice": "adservice",
    "cart": "cartservice",
    "checkout": "checkoutservice",
    "currency": "currencyservice",
    "email": "emailservice",
    "payment": "paymentservice",
    "product": "productcatalogservice",
    "catalog": "productcatalogservice",
    "recommend": "recommendationservice",
    "shipping": "shippingservice",
    "quote": "shippingservice",
    "verifycode": "ts-verification-code-service",
    "image code": "ts-verification-code-service",
    "contacts": "ts-contacts-service",
    "route": "ts-route-service",
    "basic": "ts-basic-service",
    "price": "ts-price-service",
    "seat": "ts-seat-service",
    "train": "ts-train-service",
    "travel2": "ts-travel2-service",
    "travel": "ts-travel-service",
    "order": "ts-order-service",
}


def _clean_tpl(t: str) -> str:
    return HEX_UUID_RE.sub("<*>", t)


def clean_log_err_score(bundle: CaseBundle, service: str) -> tuple[float, float, list[str]]:
    """Compute de-noised log anomaly score: true error-line surge + genuine error templates."""
    fault = bundle.logs.get("fault", {}).get(service)
    normal = bundle.logs.get("normal", {}).get(service)
    if not fault:
        return 0.0, 0.0, []
    base_err = float(normal.get("error_lines_per_min", 0.0)) if normal else 0.0
    fault_err = float(fault.get("error_lines_per_min", 0.0))
    err_delta = max(0.0, fault_err - base_err)
    if err_delta <= 0.0:
        return 0.0, 0.0, []

    # Relative error rate increase so noisy background loggers don't dominate quiet services
    rel_err = math.log1p(err_delta / max(1.0, base_err * 0.5))

    norm_clean = {_clean_tpl(k) for k in (normal or {}).get("templates", {})}
    new_clean = [t for t in fault.get("templates", {}) if _clean_tpl(t) not in norm_clean]
    err_tpls = [t for t in new_clean if ERR_TPL_RE.search(t)]

    score = 2.8 * rel_err + 0.85 * len(err_tpls)
    return score, err_delta, err_tpls


@dataclass
class VerifierConfig:
    """Hyperparameters calibrated on the Nezha Dev split only."""

    agent_prior_weight: float = 0.55
    cpu_weight: float = 1.65
    server_lat_weight: float = 1.35
    log_err_weight: float = 1.75
    trace_weight: float = 1.40
    net_io_weight: float = 0.85
    callee_propagation_bonus: float = 1.35
    caller_symptom_penalty: float = 1.10


def _infer_verified_fault_type(
    bundle: CaseBundle,
    service: str,
    cpu_z: float,
    srv_lat_z: float,
    cli_lat_z: float,
    net_z: float,
    err_delta: float,
    err_tpls: list[str],
    trace_s: float,
) -> tuple[FaultType, str]:
    """Classify the fault type for `service` from its verified multi-modal signature."""
    joined_tpls = " ".join(err_tpls).lower()

    # 1. Strong CPU saturation / contention
    if cpu_z >= 12.0 and cpu_z >= srv_lat_z * 0.25:
        return FaultType.CPU_CONTENTION, f"cpu_z={cpu_z:.1f}"

    # 2. Moderate CPU with huge trace latency slowdown (typical of cpu_consumed in Nezha OB)
    if 2.0 <= cpu_z < 12.0 and trace_s >= 2.0 and err_delta < 1.0:
        return FaultType.CPU_CONSUMED, f"cpu_z={cpu_z:.1f}, trace_slowdown={trace_s:.2f}"

    # 3. Genuine error log surge or new error templates -> distinguish return vs exception
    if err_delta >= 0.8 or err_tpls:
        if any(k in joined_tpls for k in ("warn", "not found", "code not found", "null", "empty", "false", "price", "discount")):
            return FaultType.RETURN, f"return-fault log pattern (err_delta={err_delta:.1f}/min)"
        if any(k in joined_tpls for k in EXCEPTION_KEYWORDS):
            return FaultType.EXCEPTION, f"exception log pattern (err_delta={err_delta:.1f}/min)"
        # In Online Boutique / Train-Ticket, moderate error surges without servlet exception stacktrace are often return faults
        if 0.5 <= err_delta <= 14.0 and "exception" not in joined_tpls:
            return FaultType.RETURN, f"error surge err_delta={err_delta:.1f}/min without exception stacktrace"
        return FaultType.EXCEPTION, f"error surge err_delta={err_delta:.1f}/min"

    # 4. High server latency, client latency, trace slowdown, or network I/O spike without CPU/error surge
    if srv_lat_z >= 2.0 or trace_s >= 0.25 or cli_lat_z >= 2.5 or net_z >= 3.0:
        if srv_lat_z >= 30.0 and trace_s < 0.15:
            return FaultType.EXCEPTION, f"sharp server latency spike srv_lat_z={srv_lat_z:.1f}"
        return FaultType.NETWORK_DELAY, f"latency/net anomaly (srv_z={srv_lat_z:.1f}, trace={trace_s:.2f}, net_z={net_z:.1f})"

    if cpu_z >= 2.2:
        return FaultType.CPU_CONSUMED, f"moderate cpu_z={cpu_z:.1f}"
    return FaultType.NETWORK_DELAY, "residual latency/traffic deviation"


class CounterfactualVerifier:
    """Re-evaluates candidate root causes using de-noised multi-modal evidence and call-graph propagation."""

    def __init__(self, config: VerifierConfig | None = None):
        self.config = config or VerifierConfig()

    def __call__(
        self, bundle: CaseBundle, diagnosis: Diagnosis, state: AgentState | None = None
    ) -> Diagnosis:
        cfg = self.config
        services = bundle.case.services
        if not services:
            return diagnosis

        edges = bundle.case.call_edges
        callees_of: dict[str, list[str]] = {s: [] for s in services}
        callers_of: dict[str, list[str]] = {s: [] for s in services}
        for u, v in edges:
            if u in callees_of and v in callers_of and u != v:
                callees_of[u].append(v)
                callers_of[v].append(u)

        # 1. Gather per-service de-noised multi-modal signals
        signals: dict[str, dict[str, float]] = {}
        err_deltas: dict[str, float] = {}
        err_templates: dict[str, list[str]] = {}

        for s in services:
            log_sc, err_d, err_t = clean_log_err_score(bundle, s)
            err_deltas[s] = err_d
            err_templates[s] = err_t
            cpu_z = metric_score(bundle, s, KEY_METRICS["cpu"])
            srv_z = metric_score(bundle, s, KEY_METRICS["server_latency_p90"])
            cli_z = metric_score(bundle, s, KEY_METRICS["client_latency_p90"])
            net_z = max(
                metric_score(bundle, s, KEY_METRICS["net_rx"]),
                metric_score(bundle, s, KEY_METRICS["net_tx"]),
            )
            tr_s = trace_anomaly_score(bundle, s)
            tr_drop = trace_volume_drop_score(bundle, s)

            signals[s] = {
                "cpu": cpu_z,
                "srv_lat": srv_z,
                "cli_lat": cli_z,
                "net": net_z,
                "log_err": log_sc,
                "trace": tr_s + 0.35 * tr_drop,
            }

        # Check if there is a dominant CPU contention spike in the system (suppresses background noise in large systems)
        max_cpu_z = max((signals[s]["cpu"] for s in services), default=0.0)
        max_err_d = max(err_deltas.values(), default=0.0)
        has_true_cpu_fault = any(
            signals[s]["cpu"] >= 40.0 and (signals[s]["srv_lat"] >= 8.0 or signals[s]["trace"] >= 0.4)
            for s in services
        )

        # 2. Compute weighted reciprocal-rank + magnitude scores across clean modalities
        modality_weights = {
            "cpu": cfg.cpu_weight * (1.6 if has_true_cpu_fault else (0.35 if max_err_d >= 2.0 else 0.9)),
            "srv_lat": cfg.server_lat_weight * (0.55 if max_err_d >= 2.0 and not has_true_cpu_fault else 1.0),
            "log_err": cfg.log_err_weight * (2.2 if max_err_d >= 2.0 and not has_true_cpu_fault else 1.0),
            "trace": cfg.trace_weight,
            "net": cfg.net_io_weight * (0.5 if has_true_cpu_fault or max_err_d >= 2.0 else 1.0),
        }
        thresholds = {"cpu": 1.8, "srv_lat": 1.2, "log_err": 0.45, "trace": 0.18, "net": 2.2}

        rrf_scores: dict[str, float] = {s: 0.0 for s in services}
        best_mod: dict[str, tuple[float, str]] = {s: (0.0, "none") for s in services}

        for mod, w in modality_weights.items():
            thresh = thresholds[mod]
            ordered = sorted(services, key=lambda s: signals[s][mod], reverse=True)
            for rank, s in enumerate(ordered[:8], start=1):
                val = signals[s][mod]
                if val < thresh:
                    continue
                mag_boost = min(math.log1p(val) / 4.0, 1.2)
                contrib = w * ((1.0 / rank) + 0.35 * mag_boost)
                rrf_scores[s] += contrib
                if contrib > best_mod[s][0]:
                    best_mod[s] = (contrib, mod)

        # 3. Incorporate the agent's initial ranking prior
        agent_ranks = {c.service: idx for idx, c in enumerate(diagnosis.ranking, start=1)}
        combined: dict[str, float] = {}
        for s in services:
            agent_prior = cfg.agent_prior_weight * (1.0 / agent_ranks[s]) if s in agent_ranks else 0.0
            combined[s] = rrf_scores[s] + agent_prior

        # 4. Call-graph counter-evidence & template callee-hint propagation
        notes: list[str] = []
        adjustments: dict[str, float] = {s: 0.0 for s in services}

        for u in services:
            # 4a. Check if u's error templates explicitly name a downstream callee (e.g., "Failed to retrieve ads")
            tpl_text = " ".join(err_templates[u]).lower()
            if tpl_text:
                for kw, target_svc in CALLEE_HINTS.items():
                    if kw in tpl_text and target_svc in services and target_svc != u:
                        if target_svc in callees_of[u] or signals[target_svc]["srv_lat"] > 2.0 or signals[target_svc]["cpu"] > 2.0:
                            adjustments[target_svc] += cfg.callee_propagation_bonus * 1.25
                            adjustments[u] -= cfg.caller_symptom_penalty * 0.5
                            notes.append(f"log template in {u} points to callee {target_svc} ('{kw}')")

            # 4b. If both caller u and callee v have error surges or high latency, shift weight from caller u to callee v
            active_callees = [
                v
                for v in callees_of[u]
                if (
                    signals[v]["log_err"] >= 0.8
                    or signals[v]["srv_lat"] >= 3.0
                    or signals[v]["cpu"] >= 15.0
                    or signals[v]["trace"] >= 0.5
                )
            ]
            if active_callees and signals[u]["cpu"] < 10.0:
                if signals[u]["log_err"] > 0 or signals[u]["cli_lat"] > 1.5 or signals[u]["srv_lat"] > 1.5:
                    adjustments[u] -= cfg.caller_symptom_penalty
                    for v in active_callees:
                        adjustments[v] += cfg.callee_propagation_bonus / len(active_callees)
                    notes.append(
                        f"propagated symptom from caller {u} to active callee(s) {active_callees}"
                    )

        for s in services:
            combined[s] = max(0.0, combined[s] + adjustments[s])

        # 5. Build verified ranking and fault-type classifications
        ordered_services = sorted(services, key=lambda s: combined[s], reverse=True)
        max_score = max(combined.values()) if combined else 1.0
        verified_ranking: list[RankedCause] = []
        for s in ordered_services[:5]:
            sig = signals[s]
            ft, reason = _infer_verified_fault_type(
                bundle,
                s,
                sig["cpu"],
                sig["srv_lat"],
                sig["cli_lat"],
                sig["net"],
                err_deltas[s],
                err_templates[s],
                sig["trace"],
            )
            conf = round(min(max(combined[s] / max(max_score, 1e-6), 0.05), 0.99), 2)
            ev = [
                f"verified_score={combined[s]:.2f} (primary={best_mod[s][1]})",
                reason,
            ]
            if adjustments[s] != 0.0:
                ev.append(f"call_graph_adj={adjustments[s]:+.2f}")
            verified_ranking.append(
                RankedCause(service=s, fault_type=ft, confidence=conf, evidence=ev)
            )

        old_top1 = diagnosis.ranking[0].service if diagnosis.ranking else None
        new_top1 = verified_ranking[0].service if verified_ranking else None
        if old_top1 != new_top1:
            notes.insert(0, f"verifier promoted {new_top1} over {old_top1}")
        else:
            notes.insert(0, f"verifier confirmed top-1 {new_top1}")

        if state is not None:
            state.usage.verifier_calls += 1
            state.gap_events.append(
                {
                    "gap": "verifier",
                    "step": state.step,
                    "old_top1": old_top1,
                    "new_top1": new_top1,
                    "new_fault_type": verified_ranking[0].fault_type.value if verified_ranking else "unknown",
                    "notes": notes[:4],
                }
            )

        method_name = diagnosis.method if "+verifier" in diagnosis.method else f"{diagnosis.method}+verifier"
        return Diagnosis(
            case_id=diagnosis.case_id,
            method=method_name,
            ranking=verified_ranking,
            summary=diagnosis.summary or "; ".join(notes[:2]),
            verified=True,
            verification_notes=notes[:5],
        )


def verify_diagnosis(
    bundle: CaseBundle, diagnosis: Diagnosis, state: AgentState | None = None
) -> Diagnosis:
    return CounterfactualVerifier()(bundle, diagnosis, state=state)
