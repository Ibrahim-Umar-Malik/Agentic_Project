"""E3: Proposed full system (Verifier + BudgetController + IncidentMemory all ON) on held-out Test sets.

Evaluates on:
  1. Nezha Test split (71 cases, 3 seeds)
  2. Held-out RCAEval RE2/RE3 dataset (12 cases, 3 seeds)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from microrca.data.store import DEFAULT_DIR, RCAEVAL_DIR
from microrca.eval.run import run_evaluation

DEFAULT_SEEDS = [42, 43, 44]


def run_e3(
    provider: str = "scripted",
    model: str | None = None,
    seeds: list[int] | None = None,
    out_dir: Path = Path("results/e3_full_system"),
    log_dir: Path = Path("logs"),
    quiet: bool = True,
) -> dict:
    seeds = seeds or DEFAULT_SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)

    nezha_report = run_evaluation(
        method="agent",
        provider=provider,
        model=model,
        temperature=0.2,
        data_dir=DEFAULT_DIR,
        split="test",
        seeds=seeds,
        use_verifier=True,
        use_budget=True,
        use_memory=True,
        out_dir=out_dir,
        log_dir=log_dir,
        tag_override=f"E3_full_system_{provider}",
        quiet=quiet,
    )

    rcaeval_report = None
    if RCAEVAL_DIR.exists() and list(RCAEVAL_DIR.glob("*.json.gz")):
        rcaeval_report = run_evaluation(
            method="agent",
            provider=provider,
            model=model,
            temperature=0.2,
            data_dir=RCAEVAL_DIR,
            split="test",
            seeds=seeds,
            use_verifier=True,
            use_budget=True,
            use_memory=True,
            out_dir=out_dir,
            log_dir=log_dir,
            tag_override=f"E3_full_system_rcaeval_{provider}",
            quiet=quiet,
        )

    summary = {
        "experiment": "E3_proposed_full_system",
        "split": "test",
        "provider": provider,
        "is_offline_scripted": provider == "scripted",
        "seeds": seeds,
        "nezha_test": {
            "multi_seed": nezha_report["multi_seed"],
            "by_system": nezha_report["by_system"],
            "by_fault": nezha_report["by_fault"],
        },
        "rcaeval_heldout": rcaeval_report["multi_seed"] if rcaeval_report else None,
    }
    (out_dir / "e3_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", type=Path, default=Path("results/e3_full_system"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = run_e3(provider=args.provider, model=args.model, seeds=seeds, out_dir=args.out, quiet=False)
    print(json.dumps(summary["nezha_test"]["multi_seed"], indent=2))


if __name__ == "__main__":
    main()
