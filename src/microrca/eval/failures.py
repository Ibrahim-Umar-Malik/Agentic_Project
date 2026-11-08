"""T5: Failure analysis and case-study exporter for E3 test evaluations.

Categorizes every failed E3 test case into root-cause failure modes, writes category
counts to `results/failure_categories.csv`, and exports >=2 success and >=2 failure
case studies (Markdown analysis + full JSONL trajectory) to `results/case_studies/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from microrca.data.store import DEFAULT_DIR, CaseBundle, load_cases
from microrca.gaps.verifier import clean_log_err_score
from microrca.tools.telemetry import KEY_METRICS, metric_score, trace_anomaly_score


def categorize_failure(bundle: CaseBundle, record: dict[str, Any]) -> str:
    """Assign a diagnostic failure category to a missed E3 test case."""
    if record.get("error") or not record.get("predicted"):
        return "unparseable_or_execution_error"

    true_svc = record["true_service"]
    true_fault = record["true_fault"]
    predicted = record["predicted"]

    # Check if true root cause was localized within Top-3 / Top-5 but beaten at Rank-1 by a caller/neighbor
    if true_svc in predicted[1:5]:
        return "caller_callee_cascade_ambiguity"

    log_sc, err_delta, _ = clean_log_err_score(bundle, true_svc)
    cpu_z = metric_score(bundle, true_svc, KEY_METRICS["cpu"])
    srv_z = metric_score(bundle, true_svc, KEY_METRICS["server_latency_p90"])
    tr_s = trace_anomaly_score(bundle, true_svc)

    if true_fault == "network_delay" and srv_z < 2.0 and tr_s < 0.3:
        return "subtle_network_delay_propagation"
    if true_fault in ("exception", "return") and err_delta <= 0.0 and log_sc <= 0.1:
        return "silent_code_fault_no_error_logs"
    if cpu_z < 3.0 and srv_z < 3.0 and log_sc < 0.5:
        return "background_metric_noise_masking"
    return "caller_callee_cascade_ambiguity"


def analyze_failures(
    cases_jsonl: Path,
    log_dir: Path = Path("logs"),
    out_csv: Path = Path("results/failure_categories.csv"),
    case_studies_dir: Path = Path("results/case_studies"),
    seed_filter: int = 42,
) -> dict[str, Any]:
    raw_lines = [json.loads(line) for line in cases_jsonl.read_text().splitlines() if line.strip()]
    records = [r for r in raw_lines if r.get("seed", 42) == seed_filter]
    if not records:
        records = raw_lines

    bundles = {b.case.case_id: b for b in load_cases(DEFAULT_DIR)}
    hits = [r for r in records if r.get("predicted") and r["predicted"][0] == r["true_service"]]
    misses = [r for r in records if not r.get("predicted") or r["predicted"][0] != r["true_service"]]

    cat_counts: Counter[str] = Counter()
    cat_examples: dict[str, list[str]] = defaultdict(list)
    detailed_failures: list[dict[str, Any]] = []

    for m in misses:
        cid = m["case_id"]
        b = bundles.get(cid)
        cat = categorize_failure(b, m) if b else "unknown"
        cat_counts[cat] += 1
        if len(cat_examples[cat]) < 4:
            cat_examples[cat].append(cid)
        detailed_failures.append(
            {
                "case_id": cid,
                "system": m["system"],
                "true_service": m["true_service"],
                "true_fault": m["true_fault"],
                "predicted_top3": m.get("predicted", [])[:3],
                "predicted_fault": m.get("predicted_fault", "unknown"),
                "category": cat,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total_misses = max(len(misses), 1)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["category", "count", "share_pct", "example_cases"]
        )
        writer.writeheader()
        for cat, cnt in cat_counts.most_common():
            writer.writerow(
                {
                    "category": cat,
                    "count": cnt,
                    "share_pct": round(100.0 * cnt / total_misses, 1),
                    "example_cases": ";".join(cat_examples[cat]),
                }
            )

    (out_csv.parent / "failure_details.json").write_text(
        json.dumps(
            {
                "total_test_cases": len(records),
                "top1_hits": len(hits),
                "top1_misses": len(misses),
                "categories": dict(cat_counts),
                "failures": detailed_failures,
            },
            indent=2,
        )
    )

    # Export >= 2 successes and >= 2 failures with full trajectories to results/case_studies/
    case_studies_dir.mkdir(parents=True, exist_ok=True)
    exported_successes = _export_studies(hits[:3], "success", bundles, log_dir, case_studies_dir, seed_filter)
    exported_failures = _export_studies(
        misses[:3], "failure", bundles, log_dir, case_studies_dir, seed_filter, detailed_failures
    )

    return {
        "total_cases": len(records),
        "hits": len(hits),
        "misses": len(misses),
        "categories": dict(cat_counts),
        "csv": str(out_csv),
        "exported_successes": exported_successes,
        "exported_failures": exported_failures,
    }


def _find_trajectory_file(log_dir: Path, case_id: str, seed: int) -> Path | None:
    candidates = sorted(log_dir.glob(f"*E3*seed-{seed}/{case_id}.jsonl"))
    if not candidates:
        candidates = sorted(log_dir.glob(f"*seed-{seed}/{case_id}.jsonl"))
    if not candidates:
        candidates = sorted(log_dir.glob(f"*/{case_id}.jsonl"))
    return candidates[-1] if candidates else None


def _export_studies(
    selected: list[dict[str, Any]],
    kind: str,
    bundles: dict[str, CaseBundle],
    log_dir: Path,
    case_studies_dir: Path,
    seed: int,
    detailed_failures: list[dict[str, Any]] | None = None,
) -> list[str]:
    exported: list[str] = []
    fail_map = {d["case_id"]: d["category"] for d in (detailed_failures or [])}

    for idx, rec in enumerate(selected, start=1):
        cid = rec["case_id"]
        traj_src = _find_trajectory_file(log_dir, cid, seed)
        traj_dst = case_studies_dir / f"{kind}_{idx}_{cid}.jsonl"
        events: list[dict[str, Any]] = []
        if traj_src and traj_src.exists():
            shutil.copyfile(traj_src, traj_dst)
            events = [json.loads(line) for line in traj_src.read_text().splitlines() if line.strip()]
        else:
            traj_dst.write_text(json.dumps(rec) + "\n")

        gap_actions = [e for e in events if e.get("event") == "gap_action"]
        obs_events = [e for e in events if e.get("event") == "observation"]
        end_event = next((e for e in reversed(events) if e.get("event") == "run_end"), {})

        md_lines = [
            f"# Case Study ({kind.upper()} #{idx}): `{cid}`",
            "",
            f"- **System**: `{rec['system']}`",
            f"- **Ground-Truth Service**: `{rec['true_service']}`",
            f"- **Ground-Truth Fault Type**: `{rec['true_fault']}`",
            f"- **Predicted Top-5 Ranking**: `{rec.get('predicted', [])[:5]}`",
            f"- **Predicted Fault Type**: `{rec.get('predicted_fault', 'unknown')}`",
            f"- **Outcome**: **{'TOP-1 HIT' if kind == 'success' else 'TOP-1 MISS (' + fail_map.get(cid, 'missed') + ')'}**",
            f"- **Trajectory File**: `{traj_dst.name}`",
            "",
            "## Usage & Budget Summary",
            f"- LLM Calls: `{rec['usage']['llm_calls']}`",
            f"- Tool Calls: `{rec['usage']['tool_calls']}`",
            f"- Tokens: `{rec['usage']['input_tokens'] + rec['usage']['output_tokens']}`",
            f"- Early Stop Triggered: `{rec['usage'].get('early_stop', False)}`",
            f"- Verifier Calls: `{rec['usage'].get('verifier_calls', 0)}`",
            "",
            "## Tool Observations Executed",
        ]
        for obs in obs_events:
            md_lines.append(f"- Tool `{obs.get('name')}` (ok=`{obs.get('ok')}`)")
        md_lines.append("")
        md_lines.append("## Phase 2 Gap Module Actions")
        for ga in gap_actions:
            md_lines.append(f"- **{ga.get('gap')}**: `{json.dumps(ga)}`")
        if end_event.get("verification_notes"):
            md_lines.append("")
            md_lines.append("## Verifier Reasoning Notes")
            for note in end_event["verification_notes"]:
                md_lines.append(f"- {note}")

        md_path = case_studies_dir / f"{kind}_{idx}_{cid}.md"
        md_path.write_text("\n".join(md_lines) + "\n")
        exported.append(str(md_path))
    return exported


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        type=Path,
        default=Path("results/e3_full_system/E3_full_system_scripted_cases.jsonl"),
    )
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/failure_categories.csv"))
    ap.add_argument("--case-studies-dir", type=Path, default=Path("results/case_studies"))
    args = ap.parse_args()
    res = analyze_failures(
        cases_jsonl=args.cases,
        log_dir=args.log_dir,
        out_csv=args.out_csv,
        case_studies_dir=args.case_studies_dir,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
