"""T6: Auto-generate a Markdown Gap-Resolution Table comparing Base Paper (E2 OpenRCA) vs Our Result (mean ± std).

Reads the JSON outputs from `results/e2_openrca/`, `results/e4_gap_validation/`, `results/e5_ablation/`,
and `results/e3_full_system/` and produces `results/gap_resolution_table.md`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_fmt(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text())
    ms = data.get("multi_seed", {})
    return ms.get("formatted", {}), ms.get("mean", data.get("overall", {}))


def generate_gap_table(
    results_root: Path = Path("results"),
    provider: str = "scripted",
    out_md: Path = Path("results/gap_resolution_table.md"),
) -> str:
    e2_path = results_root / "e2_openrca" / f"E2_openrca_{provider}.json"
    e3_path = results_root / "e3_full_system" / f"E3_full_system_{provider}.json"
    g1_path = results_root / "e4_gap_validation" / f"E4_P0_plus_Gap1_Verifier_{provider}.json"
    g2_path = results_root / "e4_gap_validation" / f"E4_P0_plus_Gap2_Budget_{provider}.json"
    g3_path = results_root / "e4_gap_validation" / f"E4_P0_plus_Gap3_Memory_{provider}.json"

    e2_fmt, e2_mean = _load_fmt(e2_path)
    e3_fmt, e3_mean = _load_fmt(e3_path)
    g1_fmt, g1_mean = _load_fmt(g1_path)
    g2_fmt, g2_mean = _load_fmt(g2_path)
    g3_fmt, g3_mean = _load_fmt(g3_path)

    provider_note = (
        "> **Authenticity Note**: The numbers below were produced with `--provider scripted` "
        "(offline deterministic stand-in policy) to verify the end-to-end Phase 2 pipeline. "
        "Re-run `python experiments/run_all.py --provider openai --model gpt-4o-mini` (or `gemini`/`anthropic`) "
        "with a live API key in `.env` to populate live LLM numbers."
        if provider == "scripted"
        else f"> **Provider**: `{provider}` (3 seeds: 42, 43, 44 on the 71-case held-out Nezha Test split)."
    )

    lines = [
        "# Phase 2 Gap-Resolution Comparison Table (Base Paper vs. Proposed MicroRCA-Agent)",
        "",
        provider_note,
        "",
        "## 1. Per-Gap Resolution Table (Held-Out Test Split, 3 Seeds `mean ± std`)",
        "",
        "| Research Gap | Target Metric(s) | Base Paper (`E2-OpenRCA`) | Isolated Gap (`P0 + Gap`) | Our Full System (`E3-Proposed`) | Resolution Summary |",
        "|---|---|---|---|---|---|",
        (
            f"| **Gap 1: Multi-Modal Counterfactual Verifier** (`gaps/verifier.py`) | "
            f"`AC@1` / `Avg@5` / `Fault-Type Acc` | "
            f"`{e2_fmt.get('AC@1', 'N/A')}` / `{e2_fmt.get('Avg@5', 'N/A')}` / `{e2_fmt.get('fault_type_acc|top1_hit', 'N/A')}` | "
            f"`{g1_fmt.get('AC@1', 'N/A')}` / `{g1_fmt.get('Avg@5', 'N/A')}` / `{g1_fmt.get('fault_type_acc|top1_hit', 'N/A')}` | "
            f"**`{e3_fmt.get('AC@1', 'N/A')}`** / **`{e3_fmt.get('Avg@5', 'N/A')}`** / **`{e3_fmt.get('fault_type_acc|top1_hit', 'N/A')}`** | "
            f"De-noises UUID template bursts, discounts upstream caller latency symptoms, and verifies fault signatures. |"
        ),
        (
            f"| **Gap 2: Adaptive Cost-Aware Budget Controller** (`gaps/budget.py`) | "
            f"`Tool Calls` / `Tokens` / `Early-Stop Rate` | "
            f"`{e2_fmt.get('avg_tool_calls', 'N/A')}` / `{e2_fmt.get('avg_tokens', 'N/A')}` / `{e2_fmt.get('early_stop_rate', 'N/A')}` | "
            f"`{g2_fmt.get('avg_tool_calls', 'N/A')}` / `{g2_fmt.get('avg_tokens', 'N/A')}` / `{g2_fmt.get('early_stop_rate', 'N/A')}` | "
            f"**`{e3_fmt.get('avg_tool_calls', 'N/A')}`** / **`{e3_fmt.get('avg_tokens', 'N/A')}`** / **`{e3_fmt.get('early_stop_rate', 'N/A')}`** | "
            f"Caches duplicate tool queries, bounds step fan-out, and converges early once top suspects are covered. |"
        ),
        (
            f"| **Gap 3: Leakage-Free Incident Memory** (`gaps/memory.py`) | "
            f"`AC@1` / `AC@3` / `Avg@5` | "
            f"`{e2_fmt.get('AC@1', 'N/A')}` / `{e2_fmt.get('AC@3', 'N/A')}` / `{e2_fmt.get('Avg@5', 'N/A')}` | "
            f"`{g3_fmt.get('AC@1', 'N/A')}` / `{g3_fmt.get('AC@3', 'N/A')}` / `{g3_fmt.get('Avg@5', 'N/A')}` | "
            f"**`{e3_fmt.get('AC@1', 'N/A')}`** / **`{e3_fmt.get('AC@3', 'N/A')}`** / **`{e3_fmt.get('Avg@5', 'N/A')}`** | "
            f"Retrieves Dev-only historical telemetry fingerprints via cosine similarity with zero test-set label leakage. |"
        ),
        "",
        "## 2. Overall Localization & Efficiency Comparison (`E2-OpenRCA` vs. `E3-Proposed`)",
        "",
        "| Metric | Base Paper (`E2-OpenRCA`) | Proposed Full System (`E3`) | Delta (`Our - Base`) |",
        "|---|---|---|---|",
    ]

    metrics_order = [
        ("AC@1", "AC@1", 3),
        ("AC@3", "AC@3", 3),
        ("AC@5", "AC@5", 3),
        ("Avg@5", "Avg@5", 3),
        ("Fault-Type Accuracy (Top-1 Hit)", "fault_type_acc|top1_hit", 3),
        ("Avg LLM Calls", "avg_llm_calls", 2),
        ("Avg Tool Calls", "avg_tool_calls", 2),
        ("Avg Tokens", "avg_tokens", 1),
        ("Avg Cost (USD)", "avg_cost_usd", 6),
        ("Early-Stop Rate", "early_stop_rate", 3),
    ]
    for label, key, dec in metrics_order:
        b_val = float(e2_mean.get(key, 0.0))
        o_val = float(e3_mean.get(key, 0.0))
        delta = o_val - b_val
        lines.append(
            f"| {label} | `{e2_fmt.get(key, 'N/A')}` | **`{e3_fmt.get(key, 'N/A')}`** | `{delta:+.{dec}f}` |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    out_md.write_text(content)
    return content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--out", type=Path, default=Path("results/gap_resolution_table.md"))
    args = ap.parse_args()
    md = generate_gap_table(results_root=args.results_root, provider=args.provider, out_md=args.out)
    print(md)


if __name__ == "__main__":
    main()
