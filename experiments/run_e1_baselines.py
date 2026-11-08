"""E1: Baselines (B0-stat-metric, B0-stat-multimodal, B1-single-call) on the held-out Test split.

Runs 3 seeds (default: 42, 43, 44) and writes JSON + JSONL results to `results/e1_baselines/`.
Note: When `--provider scripted` is used, outputs are marked `"is_offline_scripted": true`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from microrca.eval.run import run_evaluation

DEFAULT_SEEDS = [42, 43, 44]


def run_e1(
    provider: str = "scripted",
    model: str | None = None,
    seeds: list[int] | None = None,
    out_dir: Path = Path("results/e1_baselines"),
    log_dir: Path = Path("logs"),
    quiet: bool = True,
) -> dict:
    seeds = seeds or DEFAULT_SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)

    r_stat = run_evaluation(
        method="stat",
        split="test",
        seeds=[seeds[0]],
        out_dir=out_dir,
        log_dir=log_dir,
        tag_override="E1_B0-stat-metric",
        quiet=quiet,
    )
    r_stat_mm = run_evaluation(
        method="stat-mm",
        split="test",
        seeds=[seeds[0]],
        out_dir=out_dir,
        log_dir=log_dir,
        tag_override="E1_B0-stat-multimodal",
        quiet=quiet,
    )
    r_b1 = run_evaluation(
        method="single",
        provider=provider,
        model=model,
        temperature=0.2,
        split="test",
        seeds=seeds,
        out_dir=out_dir,
        log_dir=log_dir,
        tag_override=f"E1_B1-single-call_{provider}",
        quiet=quiet,
    )

    summary = {
        "experiment": "E1_baselines",
        "split": "test",
        "provider": provider,
        "is_offline_scripted": provider == "scripted",
        "seeds": seeds,
        "methods": {
            "B0-stat-metric": r_stat["multi_seed"],
            "B0-stat-multimodal": r_stat_mm["multi_seed"],
            f"B1-single-call:{provider}": r_b1["multi_seed"],
        },
    }
    (out_dir / "e1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", type=Path, default=Path("results/e1_baselines"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = run_e1(provider=args.provider, model=args.model, seeds=seeds, out_dir=args.out, quiet=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
