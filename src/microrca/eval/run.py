"""Run one RCA method over the evaluation set and write per-case records plus a summary.

Examples
  python -m microrca.eval.run --method stat                       # B0, metrics only
  python -m microrca.eval.run --method stat-mm                    # B0, metrics+logs+traces
  python -m microrca.eval.run --method single --provider openai   # B1, one LLM call
  python -m microrca.eval.run --method agent --provider anthropic --model claude-haiku-4-5
  python -m microrca.eval.run --method agent --provider scripted  # offline end-to-end smoke run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from microrca.baselines import single_call, statistical
from microrca.agent import react_agent
from microrca.contracts import EvalRecord, Usage
from microrca.data.store import DEFAULT_DIR, load_cases
from microrca.eval.metrics import by_group, summarize
from microrca.llm.providers import get_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["stat", "stat-mm", "single", "agent"], required=True)
    ap.add_argument("--provider", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--system", choices=["online-boutique", "train-ticket"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    cases = load_cases(args.data, args.system, args.limit)
    llm = get_model(args.provider, args.model) if args.method in {"single", "agent"} else None
    records = []
    for bundle in cases:
        t0 = time.time()
        if args.method in {"stat", "stat-mm"}:
            diag, usage = statistical.diagnose(bundle, multimodal=args.method == "stat-mm"), Usage()
            usage.latency_s = time.time() - t0
        elif args.method == "single":
            diag, usage = single_call.diagnose(bundle, llm)
        else:
            diag, state = react_agent.run(bundle, llm)
            usage = state.usage
        label = bundle.case.label
        records.append(EvalRecord(
            case_id=bundle.case.case_id, system=bundle.case.system, method=diag.method,
            true_service=label.service, true_fault=label.fault_type.value,
            predicted=[c.service for c in diag.ranking],
            predicted_fault=diag.ranking[0].fault_type.value if diag.ranking else "unknown",
            usage=usage))
        hit = "HIT " if records[-1].predicted[:1] == [label.service] else "miss"
        print(f"[{hit}] {bundle.case.case_id:22s} true={label.service:32s} pred={records[-1].predicted[:3]}")

    method = records[0].method if records else args.method
    tag = method.replace(":", "_").replace("/", "_")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"method": method, "overall": summarize(records), "by_system": by_group(records, "system"),
              "by_fault": by_group(records, "true_fault")}
    (args.out / f"{tag}.json").write_text(json.dumps(report, indent=2))
    with (args.out / f"{tag}_cases.jsonl").open("w") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote {args.out / f'{tag}.json'}")


if __name__ == "__main__":
    main()
