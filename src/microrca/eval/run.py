"""Run one RCA method over the evaluation set and write per-case records plus a summary.

Examples
  python -m microrca.eval.run --method stat --split test
  python -m microrca.eval.run --method stat-mm --split test
  python -m microrca.eval.run --method single --provider openai --split test
  python -m microrca.eval.run --method openrca --provider scripted --split test
  python -m microrca.eval.run --method agent --all-gaps --provider scripted --split test --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from microrca.agent import react_agent
from microrca.baselines import openrca, single_call, statistical
from microrca.contracts import EvalRecord, Usage
from microrca.data.store import DEFAULT_DIR, RCAEVAL_DIR, load_cases
from microrca.eval.metrics import aggregate_seed_summaries, by_group, summarize
from microrca.gaps.budget import BudgetController
from microrca.gaps.memory import IncidentMemory
from microrca.gaps.verifier import CounterfactualVerifier
from microrca.llm.providers import get_model


def build_method_tag(
    method: str,
    llm_name: str | None,
    use_verifier: bool,
    use_budget: bool,
    use_memory: bool,
) -> str:
    if method == "stat":
        return "B0-stat-metric"
    if method == "stat-mm":
        return "B0-stat-multimodal"
    if method == "single":
        return f"B1-single-call:{llm_name}"
    if method == "openrca":
        return f"E2-openrca:{llm_name}"
    if use_verifier and use_budget and use_memory:
        return f"P1-full-agent:{llm_name}"
    if not (use_verifier or use_budget or use_memory):
        return f"P0-react-agent:{llm_name}"
    parts = []
    if use_verifier:
        parts.append("verifier")
    if use_budget:
        parts.append("budget")
    if use_memory:
        parts.append("memory")
    return f"P1-agent({'+'.join(parts)}):{llm_name}"


def run_evaluation(
    method: str,
    provider: str = "scripted",
    model: str | None = None,
    temperature: float = 0.0,
    system: str | None = None,
    limit: int | None = None,
    data_dir: Path = DEFAULT_DIR,
    split: str = "all",
    seeds: list[int] | None = None,
    use_verifier: bool = False,
    use_budget: bool = False,
    use_memory: bool = False,
    out_dir: Path = Path("results"),
    log_dir: Path = Path("logs"),
    tag_override: str | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    seeds = seeds or [42]
    cases = load_cases(data_dir, system=system, limit=limit, split=split)
    dev_cases = load_cases(DEFAULT_DIR, split="dev") if use_memory else []

    all_records: list[EvalRecord] = []
    seed_summaries: list[dict[str, float]] = []

    for seed in seeds:
        llm = (
            get_model(provider, model, temperature=temperature, seed=seed)
            if method in {"single", "openrca", "agent"}
            else None
        )
        llm_name = llm.name if llm else None
        m_tag = tag_override or build_method_tag(method, llm_name, use_verifier, use_budget, use_memory)
        safe_tag = m_tag.replace(":", "_").replace("/", "_")
        run_id = f"{safe_tag}_split-{split}_seed-{seed}"

        mem = IncidentMemory.from_dev_cases(dev_cases) if use_memory else None
        if mem is not None:
            mem.assert_no_test_leakage()
        ver = CounterfactualVerifier() if use_verifier else None

        seed_records: list[EvalRecord] = []
        for bundle in cases:
            t0 = time.time()
            err_msg: str | None = None
            if method in {"stat", "stat-mm"}:
                diag = statistical.diagnose(bundle.blind_bundle(), multimodal=(method == "stat-mm"))
                usage = Usage(latency_s=time.time() - t0)
            elif method == "single":
                diag, usage = single_call.diagnose(bundle, llm)
            elif method == "openrca":
                diag, state = openrca.diagnose(bundle, llm, seed=seed, run_id=run_id, log_dir=log_dir)
                usage, err_msg = state.usage, state.error
            else:
                b_ctrl = BudgetController() if use_budget else None
                diag, state = react_agent.run(
                    bundle,
                    llm,
                    memory=mem,
                    verifier=ver,
                    budget_controller=b_ctrl,
                    seed=seed,
                    run_id=run_id,
                    log_dir=log_dir,
                    method_tag=m_tag,
                )
                usage, err_msg = state.usage, state.error

            label = bundle.case.label
            rec = EvalRecord(
                case_id=bundle.case.case_id,
                system=bundle.case.system,
                method=m_tag,
                true_service=label.service,
                true_fault=label.fault_type.value,
                predicted=[c.service for c in diag.ranking],
                predicted_fault=diag.ranking[0].fault_type.value if diag.ranking else "unknown",
                usage=usage,
                seed=seed,
                split=split,
                error=err_msg,
            )
            seed_records.append(rec)
            all_records.append(rec)
            if not quiet:
                hit = "HIT " if rec.predicted[:1] == [label.service] else "miss"
                print(
                    f"[{hit}] (seed={seed}) {bundle.case.case_id:22s} "
                    f"true={label.service:28s} pred={rec.predicted[:3]}"
                )

        if mem is not None:
            mem.assert_no_test_leakage()
        seed_summaries.append(summarize(seed_records))

    final_method = all_records[0].method if all_records else (tag_override or method)
    file_tag = (tag_override or final_method).replace(":", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_seed_summaries(seed_summaries)

    report = {
        "method": final_method,
        "provider": provider,
        "is_offline_scripted": provider == "scripted",
        "split": split,
        "seeds": seeds,
        "gaps": {"verifier": use_verifier, "budget": use_budget, "memory": use_memory},
        "overall": agg["mean"] if len(seeds) > 1 else summarize(all_records),
        "multi_seed": agg,
        "per_seed": {str(s): summ for s, summ in zip(seeds, seed_summaries)},
        "by_system": by_group(all_records, "system"),
        "by_fault": by_group(all_records, "true_fault"),
    }
    (out_dir / f"{file_tag}.json").write_text(json.dumps(report, indent=2))
    with (out_dir / f"{file_tag}_cases.jsonl").open("w") as fh:
        for r in all_records:
            fh.write(r.model_dump_json() + "\n")
    if not quiet:
        print(json.dumps(report["overall"], indent=2))
        print(f"wrote {out_dir / f'{file_tag}.json'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["stat", "stat-mm", "single", "openrca", "agent"], required=True)
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--system", choices=["online-boutique", "train-ticket", "sock-shop"], default=None)
    ap.add_argument("--dataset", choices=["nezha", "rcaeval"], default="nezha")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--seeds", default="42", help="Comma-separated random seeds, e.g. 42,43,44")
    ap.add_argument("--verifier", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--budget", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--memory", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--all-gaps", action="store_true", help="Enable verifier, budget, and memory")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    data_dir = args.data or (RCAEVAL_DIR if args.dataset == "rcaeval" else DEFAULT_DIR)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    use_v = args.verifier or args.all_gaps
    use_b = args.budget or args.all_gaps
    use_m = args.memory or args.all_gaps

    run_evaluation(
        method=args.method,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        system=args.system,
        limit=args.limit,
        data_dir=data_dir,
        split=args.split,
        seeds=seeds,
        use_verifier=use_v,
        use_budget=use_b,
        use_memory=use_m,
        out_dir=args.out,
        log_dir=args.log_dir,
        tag_override=args.tag,
    )


if __name__ == "__main__":
    main()
