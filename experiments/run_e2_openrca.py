"""E2: Base paper (OpenRCA, Xu et al., ICLR 2025) reproduction on our test split and metrics.

Runs the unverified multi-step telemetry query agent over 3 seeds (42, 43, 44).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from microrca.eval.run import run_evaluation

DEFAULT_SEEDS = [42, 43, 44]


def run_e2(
    provider: str = "scripted",
    model: str | None = None,
    seeds: list[int] | None = None,
    out_dir: Path = Path("results/e2_openrca"),
    log_dir: Path = Path("logs"),
    quiet: bool = True,
) -> dict:
    seeds = seeds or DEFAULT_SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_evaluation(
        method="openrca",
        provider=provider,
        model=model,
        temperature=0.2,
        split="test",
        seeds=seeds,
        use_verifier=False,
        use_budget=False,
        use_memory=False,
        out_dir=out_dir,
        log_dir=log_dir,
        tag_override=f"E2_openrca_{provider}",
        quiet=quiet,
    )
    summary = {
        "experiment": "E2_openrca_reproduction",
        "split": "test",
        "provider": provider,
        "is_offline_scripted": provider == "scripted",
        "seeds": seeds,
        "multi_seed": report["multi_seed"],
        "by_system": report["by_system"],
        "by_fault": report["by_fault"],
    }
    (out_dir / "e2_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", type=Path, default=Path("results/e2_openrca"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = run_e2(provider=args.provider, model=args.model, seeds=seeds, out_dir=args.out, quiet=False)
    print(json.dumps(summary["multi_seed"], indent=2))


if __name__ == "__main__":
    main()
