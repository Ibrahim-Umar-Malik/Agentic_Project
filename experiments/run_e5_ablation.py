"""E5: Ablation study (remove one gap at a time from the proposed full system) across 3 seeds.

Variants evaluated on the held-out Test split:
  - Full_all_gaps       (Verifier=ON,  Budget=ON,  Memory=ON)
  - Full_minus_Verifier (Verifier=OFF, Budget=ON,  Memory=ON)
  - Full_minus_Budget   (Verifier=ON,  Budget=OFF, Memory=ON)
  - Full_minus_Memory   (Verifier=ON,  Budget=ON,  Memory=OFF)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from microrca.eval.run import run_evaluation

DEFAULT_SEEDS = [42, 43, 44]


def run_e5(
    provider: str = "scripted",
    model: str | None = None,
    seeds: list[int] | None = None,
    out_dir: Path = Path("results/e5_ablation"),
    log_dir: Path = Path("logs"),
    quiet: bool = True,
) -> dict:
    seeds = seeds or DEFAULT_SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("Full_all_gaps", True, True, True),
        ("Full_minus_Verifier", False, True, True),
        ("Full_minus_Budget", True, False, True),
        ("Full_minus_Memory", True, True, False),
    ]
    ablations = {}
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
            tag_override=f"E5_{label}_{provider}",
            quiet=quiet,
        )
        ablations[label] = rep["multi_seed"]

    summary = {
        "experiment": "E5_ablation",
        "split": "test",
        "provider": provider,
        "is_offline_scripted": provider == "scripted",
        "seeds": seeds,
        "ablations": ablations,
    }
    (out_dir / "e5_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", type=Path, default=Path("results/e5_ablation"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = run_e5(provider=args.provider, model=args.model, seeds=seeds, out_dir=args.out, quiet=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
