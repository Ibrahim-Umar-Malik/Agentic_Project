"""Master runner for Phase 2 experiments E1 through E6, Failure Analysis (T5), and Gap-Resolution Table (T6).

Usage:
  python experiments/run_all.py --provider scripted
  python experiments/run_all.py --provider openai --model gpt-4o-mini --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e1_baselines import run_e1
from experiments.run_e2_openrca import run_e2
from experiments.run_e3_full_system import run_e3
from experiments.run_e4_gap_validation import run_e4
from experiments.run_e5_ablation import run_e5
from experiments.run_e6_efficiency import run_e6
from microrca.eval.failures import analyze_failures
from microrca.eval.gap_table import generate_gap_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    res_root = args.results_root
    prov = args.provider

    print(f"=== Running E1: Baselines (provider={prov}, seeds={seeds}) ===")
    e1 = run_e1(provider=prov, model=args.model, seeds=seeds, out_dir=res_root / "e1_baselines", log_dir=args.log_dir)

    print(f"=== Running E2: Base Paper OpenRCA Reproduction (provider={prov}, seeds={seeds}) ===")
    e2 = run_e2(provider=prov, model=args.model, seeds=seeds, out_dir=res_root / "e2_openrca", log_dir=args.log_dir)

    print(f"=== Running E3: Proposed Full System (provider={prov}, seeds={seeds}) ===")
    e3 = run_e3(provider=prov, model=args.model, seeds=seeds, out_dir=res_root / "e3_full_system", log_dir=args.log_dir)

    print(f"=== Running E4: Gap-Wise Validation (provider={prov}, seeds={seeds}) ===")
    e4 = run_e4(
        provider=prov, model=args.model, seeds=seeds, out_dir=res_root / "e4_gap_validation", log_dir=args.log_dir
    )

    print(f"=== Running E5: Ablation Study (provider={prov}, seeds={seeds}) ===")
    e5 = run_e5(provider=prov, model=args.model, seeds=seeds, out_dir=res_root / "e5_ablation", log_dir=args.log_dir)

    print("=== Running E6: Efficiency Analysis ===")
    e6 = run_e6(results_root=res_root, provider=prov, out_dir=res_root / "e6_efficiency")

    print("=== Running T5: Failure Analysis & Case Study Export ===")
    e3_cases_path = res_root / "e3_full_system" / f"E3_full_system_{prov}_cases.jsonl"
    failures_report = analyze_failures(
        cases_jsonl=e3_cases_path,
        log_dir=args.log_dir,
        out_csv=res_root / "failure_categories.csv",
        case_studies_dir=res_root / "case_studies",
        seed_filter=seeds[0],
    )

    print("=== Running T6: Auto-Generating Gap-Resolution Table ===")
    table_md = generate_gap_table(
        results_root=res_root,
        provider=prov,
        out_md=res_root / "gap_resolution_table.md",
    )
    print(table_md)

    master = {
        "provider": prov,
        "is_offline_scripted": prov == "scripted",
        "seeds": seeds,
        "E1": e1["methods"],
        "E2": e2["multi_seed"],
        "E3": e3["nezha_test"]["multi_seed"],
        "E3_rcaeval": e3["rcaeval_heldout"],
        "E4": e4["variants"],
        "E5": e5["ablations"],
        "E6_rows": len(e6["rows"]),
        "T5_failures": failures_report,
    }
    (res_root / f"phase2_master_summary_{prov}.json").write_text(json.dumps(master, indent=2))
    print(f"wrote master summary to {res_root / f'phase2_master_summary_{prov}.json'}")


if __name__ == "__main__":
    main()
