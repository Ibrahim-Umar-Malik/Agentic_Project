"""E6: Efficiency analysis (cost, latency, tokens, tool calls, LLM calls, retries, early-stop rate).

Aggregates efficiency metrics across B0, B1, E2 (OpenRCA), P0 (base agent), and E3 (Proposed full system).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EFFICIENCY_COLS = [
    "method",
    "provider",
    "AC@1",
    "Avg@5",
    "avg_llm_calls",
    "avg_tool_calls",
    "avg_tokens",
    "avg_latency_s",
    "avg_retries",
    "avg_cost_usd",
    "early_stop_rate",
]


def run_e6(
    results_root: Path = Path("results"),
    provider: str = "scripted",
    out_dir: Path = Path("results/e6_efficiency"),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("B0-stat-metric", results_root / "e1_baselines" / "E1_B0-stat-metric.json"),
        ("B0-stat-multimodal", results_root / "e1_baselines" / "E1_B0-stat-multimodal.json"),
        (f"B1-single-call ({provider})", results_root / "e1_baselines" / f"E1_B1-single-call_{provider}.json"),
        (f"E2-OpenRCA ({provider})", results_root / "e2_openrca" / f"E2_openrca_{provider}.json"),
        (f"P0-ReAct-Base ({provider})", results_root / "e4_gap_validation" / f"E4_P0_base_{provider}.json"),
        (f"E3-Full-System ({provider})", results_root / "e3_full_system" / f"E3_full_system_{provider}.json"),
    ]

    rows = []
    for label, path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        fmt = data.get("multi_seed", {}).get("formatted", {})
        mean = data.get("multi_seed", {}).get("mean", data.get("overall", {}))
        row = {
            "method": label,
            "provider": data.get("provider", "stat"),
            "AC@1": fmt.get("AC@1", str(mean.get("AC@1", ""))),
            "Avg@5": fmt.get("Avg@5", str(mean.get("Avg@5", ""))),
            "avg_llm_calls": fmt.get("avg_llm_calls", str(mean.get("avg_llm_calls", ""))),
            "avg_tool_calls": fmt.get("avg_tool_calls", str(mean.get("avg_tool_calls", ""))),
            "avg_tokens": fmt.get("avg_tokens", str(mean.get("avg_tokens", ""))),
            "avg_latency_s": fmt.get("avg_latency_s", str(mean.get("avg_latency_s", ""))),
            "avg_retries": fmt.get("avg_retries", str(mean.get("avg_retries", ""))),
            "avg_cost_usd": fmt.get("avg_cost_usd", str(mean.get("avg_cost_usd", ""))),
            "early_stop_rate": fmt.get("early_stop_rate", str(mean.get("early_stop_rate", ""))),
            "raw_mean": mean,
        }
        rows.append(row)

    csv_path = out_dir / "e6_efficiency.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EFFICIENCY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "E6_efficiency",
        "provider": provider,
        "is_offline_scripted": provider == "scripted",
        "rows": rows,
    }
    (out_dir / "e6_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/e6_efficiency"))
    args = ap.parse_args()
    summary = run_e6(results_root=args.results_root, provider=args.provider, out_dir=args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
