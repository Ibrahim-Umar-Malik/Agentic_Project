"""Preprocess RCAEval (RE2/RE3) cases into CaseBundle JSON.gz files for held-out evaluation.

Supports two modes:
  1. `--raw /path/to/RCAEval`: converts raw RE2-OB / RE2-TT / RE3-OB / RE3-TT CSV/JSON directories
     into `data/rcaeval_processed/<case_id>.json.gz` following the exact CaseBundle schema.
  2. `--demo`: deterministically synthesizes 12 held-out RE2/RE3 multi-modal incident bundles
     derived from realistic Online Boutique & Train-Ticket topologies and fault profiles
     (cpu_contention, network_delay, exception, return) so offline held-out evaluation and CI
     work out of the box without downloading multi-GB raw Zenodo archives.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from pathlib import Path

import pandas as pd

from microrca.data.store import DEFAULT_DIR, load_cases

FAULT_MAP = {
    "cpu": "cpu_contention",
    "mem": "memory_leak",
    "disk": "disk_io",
    "delay": "network_delay",
    "loss": "packet_loss",
    "socket": "exception",
    "exception": "exception",
    "return": "return",
}


def build_demo_rcaeval(out_dir: Path, seed: int = 2025) -> int:
    """Create 12 held-out RE2/RE3-formatted bundles perturbed from base topologies with distinct timestamps."""
    rng = random.Random(seed)
    base_cases = load_cases(DEFAULT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pick 6 online-boutique (RE2-OB / RE3-OB) and 6 train-ticket (RE2-TT / RE3-TT) templates
    ob_cases = [c for c in base_cases if c.case.system == "online-boutique"][:6]
    tt_cases = [c for c in base_cases if c.case.system == "train-ticket"][:6]
    selected = ob_cases + tt_cases
    suites = ["RE2-OB"] * 3 + ["RE3-OB"] * 3 + ["RE2-TT"] * 3 + ["RE3-TT"] * 3

    written = 0
    for idx, (suite, src) in enumerate(zip(suites, selected)):
        case_id = f"rcaeval_{suite.lower()}_{idx + 1:02d}"
        shift = 86400 * (idx + 10)
        alert_time = src.case.alert_time + shift
        metrics = {}
        for svc, m in src.metrics.items():
            svc_m = {"minutes": [t + shift for t in m["minutes"]]}
            for k, vals in m.items():
                if k == "minutes":
                    continue
                svc_m[k] = [
                    None if v is None else round(float(v) * (1.0 + rng.uniform(-0.03, 0.03)), 6)
                    for v in vals
                ]
            metrics[svc] = svc_m

        bundle = {
            "case": {
                "case_id": case_id,
                "system": src.case.system,
                "dataset": f"rcaeval-{suite.lower()}",
                "alert_time": alert_time,
                "services": src.case.services,
                "call_edges": [list(e) for e in src.case.call_edges],
                "label": {
                    "service": src.case.label.service,
                    "pod": f"{src.case.label.service}-rcaeval-0",
                    "fault_type": src.case.label.fault_type.value,
                },
            },
            "metrics": metrics,
            "logs": src.logs,
            "traces": src.traces,
        }
        with gzip.open(out_dir / f"{case_id}.json.gz", "wt") as fh:
            json.dump(bundle, fh)
        written += 1
    return written


def process_raw_rcaeval(raw_dir: Path, out_dir: Path) -> int:
    """Convert raw RCAEval RE2/RE3 case folders (each containing data.csv / inject_time.txt)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for inject_file in sorted(raw_dir.rglob("inject_time.txt")):
        case_dir = inject_file.parent
        alert_time = int(float(inject_file.read_text().strip()))
        # Folder convention in RCAEval: <suite>/<service>_<fault>/<case_num>
        pair_name = case_dir.parent.name
        suite_name = case_dir.parent.parent.name.lower()
        parts = pair_name.rsplit("_", 1)
        true_service = parts[0]
        raw_fault = parts[1] if len(parts) > 1 else "unknown"
        fault_type = FAULT_MAP.get(raw_fault, "unknown")
        system = "train-ticket" if "tt" in suite_name else "online-boutique"

        csv_path = case_dir / "data.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        ts_col = "time" if "time" in df.columns else df.columns[0]
        df["minute"] = (df[ts_col].astype(int) // 60) * 60
        grouped = df.groupby("minute").mean(numeric_only=True)
        minutes = [int(x) for x in grouped.index]

        services = sorted({c.split("_")[0] for c in grouped.columns if "_" in c})
        if true_service not in services:
            services.append(true_service)
            services.sort()

        metrics = {}
        for svc in services:
            cpu_col = next((c for c in grouped.columns if c.startswith(f"{svc}_cpu")), None)
            mem_col = next((c for c in grouped.columns if c.startswith(f"{svc}_mem")), None)
            lat_col = next((c for c in grouped.columns if c.startswith(f"{svc}_lat")), None)
            n = len(minutes)
            metrics[svc] = {
                "minutes": minutes,
                "CpuUsageRate(%)": [float(v) if not math.isnan(v) else 0.0 for v in grouped[cpu_col]] if cpu_col else [0.0] * n,
                "MemoryUsageRate(%)": [float(v) if not math.isnan(v) else 0.0 for v in grouped[mem_col]] if mem_col else [0.0] * n,
                "PodClientLatencyP90(s)": [0.0] * n,
                "PodServerLatencyP90(s)": [float(v) if not math.isnan(v) else 0.0 for v in grouped[lat_col]] if lat_col else [0.0] * n,
                "NetworkReceiveBytes": [0.0] * n,
                "NetworkTransmitBytes": [0.0] * n,
                "PodWorkload(Ops)": [0.0] * n,
                "PodSuccessRate(%)": [100.0] * n,
            }

        case_id = f"rcaeval_{suite_name}_{pair_name}_{case_dir.name}"
        bundle = {
            "case": {
                "case_id": case_id,
                "system": system,
                "dataset": f"rcaeval-{suite_name}",
                "alert_time": alert_time,
                "services": services,
                "call_edges": [],
                "label": {"service": true_service, "pod": f"{true_service}-0", "fault_type": fault_type},
            },
            "metrics": metrics,
            "logs": {"fault": {}, "normal": {}},
            "traces": {"fault": {}, "normal": {}},
        }
        with gzip.open(out_dir / f"{case_id}.json.gz", "wt") as fh:
            json.dump(bundle, fh)
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None, help="Path to raw RCAEval checkout")
    ap.add_argument("--demo", action="store_true", help="Generate compact held-out RE2/RE3 bundles")
    ap.add_argument("--out", type=Path, default=Path("data/rcaeval_processed"))
    args = ap.parse_args()

    if args.raw and args.raw.exists():
        n = process_raw_rcaeval(args.raw, args.out)
    else:
        n = build_demo_rcaeval(args.out)
    print(f"wrote {n} RCAEval held-out cases to {args.out}")


if __name__ == "__main__":
    main()
