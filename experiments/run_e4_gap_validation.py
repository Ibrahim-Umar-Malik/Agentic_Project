"""E4: Gap-wise validation on the held-out Test split across 3 seeds.

Validates each individual gap added on top of the base P0 ReAct agent:
  - P0 (no gaps)
  - P0 + Gap 1 (Verifier only)
  - P0 + Gap 2 (BudgetController only)
  - P0 + Gap 3 (IncidentMemory only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from microrca.eval.run import run_evaluation

DEFAULT_SEEDS = [42, 43, 44]


def run_e4(
    provider: str = "scripted",
    model: str | None = None,
    seeds: list[int] | None = None,
    out_dir: Path = Path("results/e4_gap_validation"),
    log_dir: Path = Path("logs"),
    quiet: bool = True,
) -> dict:
    seeds = seeds or DEFAULT_SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("P0_base", False, False, False),
        ("P0_plus_Gap1_Verifier", True, False, False),
        ("P0_plus_Gap2_Budget", False, True, False),
        ("P0_plus_Gap3_Memory", False, False, True),
    ]
    variants = {}
    for label, v, b, m in configs:
        rep = run_evaluation(
            method="agent",
            provider=provider,
            model=model,
            temperature=0.2,
            split="test",
            seeds=seeds,
            use_verifier=v,
            use_budget=b,
            use_memory=m,
            out_dir=out_dir,
            log_dir=log_dir,
            tag_override=f"E4_{label}_{provider}",
            quiet=quiet,
        )
        variants[label] = rep["multi_seed"]

    summary = {
        "experiment": "E4_gap_wise_validation",
        "split": "test",
        "provider": provider,
        "is_offline_scripted": provider == "scripted",
        "seeds": seeds,
        "variants": variants,
    }
    (out_dir / "e4_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", type=Path, default=Path("results/e4_gap_validation"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = run_e4(provider=args.provider, model=args.model, seeds=seeds, out_dir=args.out, quiet=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
