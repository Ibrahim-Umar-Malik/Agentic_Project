"""Convert the raw Nezha multi-modal fault dataset into compact per-case JSON bundles.

Raw data (~3.2 GB, MIT licence) comes from https://github.com/IntelligentDDS/Nezha
(fetch it with scripts/fetch_nezha.sh). Output: data/processed/<case_id>.json.gz,
one file per labelled fault case, holding only what the RCA tools need:
  * per-service metric series from 30 min before to 5 min after the alert
  * per-service log volume, error-like lines and new log templates vs. fault-free logs
  * per-service span latency statistics vs. fault-free traces
  * the service call graph and the ground-truth label

Usage: python scripts/preprocess_nezha.py --raw /path/to/Nezha --out data/processed
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEMS = {
    "2022-08-22": "online-boutique",
    "2022-08-23": "online-boutique",
    "2023-01-29": "train-ticket",
    "2023-01-30": "train-ticket",
}
METRIC_COLS = [
    "CpuUsageRate(%)", "MemoryUsageRate(%)", "SyscallRead", "SyscallWrite",
    "NetworkReceiveBytes", "NetworkTransmitBytes", "PodClientLatencyP90(s)",
    "PodServerLatencyP90(s)", "PodWorkload(Ops)", "PodSuccessRate(%)",
]
PRE_MIN, POST_MIN = 30, 5
ERR_RE = re.compile(r"error|exception|fail|refused|timeout|severity\\\":\\\"warn", re.I)
NUM_RE = re.compile(r"\b[0-9a-f]{8,}\b|\b\d+(\.\d+)?\b|[0-9a-f-]{36}", re.I)


def pod_to_service(pod: str) -> str:
    return "-".join(pod.split("-")[:-2])


def template(msg: str) -> str:
    """Cheap log templating: strip trace ids, numbers and uuids."""
    msg = re.sub(r"TraceID: \w+ SpanID: \w+", "", msg)
    msg = NUM_RE.sub("<*>", msg)
    return msg[:160].strip()


def extract_msg(raw: str) -> str:
    try:
        inner = json.loads(raw)["log"]
        try:
            return json.loads(inner).get("message", inner)
        except (json.JSONDecodeError, AttributeError):
            return inner
    except (json.JSONDecodeError, KeyError, TypeError):
        return str(raw)


def load_metrics(day_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted((day_dir / "metric").glob("*_metric.csv")):
        df = pd.read_csv(f)
        svc = pod_to_service(df["PodName"].iloc[0])
        df["minute"] = (df["TimeStamp"] // 60) * 60
        out[svc] = df.groupby("minute")[METRIC_COLS].mean()
    return out


def log_stats(files: list[Path], minutes: float) -> dict[str, dict]:
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f, usecols=["PodName", "Log"]) for f in files], ignore_index=True)
    df["service"] = df["PodName"].map(pod_to_service)
    df["msg"] = df["Log"].map(extract_msg)
    df["err"] = df["msg"].str.contains(ERR_RE)
    df["tpl"] = df["msg"].map(template)
    stats = {}
    for svc, g in df.groupby("service"):
        stats[svc] = {
            "lines_per_min": len(g) / minutes,
            "error_lines_per_min": float(g["err"].sum()) / minutes,
            "templates": g["tpl"].value_counts().head(40).to_dict(),
        }
    return stats


def trace_stats(files: list[Path]) -> dict[str, dict]:
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f, usecols=["PodName", "Duration"]) for f in files], ignore_index=True)
    df["service"] = df["PodName"].map(pod_to_service)
    out = {}
    for svc, g in df.groupby("service"):
        d = g["Duration"].to_numpy() / 1000.0  # microseconds -> ms
        out[svc] = {"spans_per_min": len(d) / max(len(files), 1), "p50_ms": float(np.percentile(d, 50)), "p90_ms": float(np.percentile(d, 90))}
    return out


def minute_files(folder: Path, suffix: str, t: int, before: int, after: int) -> list[Path]:
    files = []
    for k in range(-before, after + 1):
        ts = datetime.fromtimestamp(t + 60 * k, tz=timezone.utc)
        f = folder / f"{ts:%H_%M}_{suffix}.csv"
        if f.exists():
            files.append(f)
    return files


def call_edges(day_dir: Path, services: list[str]) -> list[tuple[str, str]]:
    dep = day_dir / "metric" / "dependency.csv"
    if dep.exists():
        d = pd.read_csv(dep)
        return sorted({(a, b) for a, b in zip(d["Source"], d["Target"])})
    # Train-Ticket has no dependency file: derive caller->callee from parent/child spans.
    edges: set[tuple[str, str]] = set()
    for f in sorted((day_dir / "trace").glob("*_trace.csv"))[:20]:
        t = pd.read_csv(f, usecols=["SpanID", "ParentID", "PodName"])
        svc = dict(zip(t["SpanID"], t["PodName"].map(pod_to_service)))
        for parent, pod in zip(t["ParentID"], t["PodName"]):
            a, b = svc.get(parent), pod_to_service(pod)
            if a and a != b and a in services and b in services:
                edges.add((a, b))
    return sorted(edges)


def process_day(raw: Path, day: str, out: Path) -> int:
    rca = raw / "rca_data" / day
    normal = raw / "construct_data" / day
    system = SYSTEMS[day]
    metrics = load_metrics(rca)
    services = sorted(metrics)
    edges = call_edges(rca, services)
    normal_logs = log_stats(sorted((normal / "log").glob("*.csv")), minutes=1)
    normal_traces = trace_stats(sorted((normal / "trace").glob("*.csv")))
    labels = json.loads((rca / f"{day}-fault_list.json").read_text())
    n = 0
    for hour, faults in labels.items():
        for i, fault in enumerate(faults):
            # inject_timestamp is wrong for some Train-Ticket rows; inject_time is UTC wall clock.
            t = int(datetime.strptime(fault["inject_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            t0, t1 = (t // 60) * 60 - PRE_MIN * 60, (t // 60) * 60 + POST_MIN * 60
            series = {}
            for svc, m in metrics.items():
                w = m.loc[(m.index >= t0) & (m.index <= t1)]
                series[svc] = {"minutes": [int(x) for x in w.index]} | {
                    c: [None if pd.isna(v) else round(float(v), 6) for v in w[c]] for c in METRIC_COLS
                }
            log_files = minute_files(rca / "log", "log", t, 0, 2)
            trace_files = minute_files(rca / "trace", "trace", t, 0, 2)
            case_id = f"{day}_{hour}_{i:02d}"
            bundle = {
                "case": {
                    "case_id": case_id,
                    "system": system,
                    "alert_time": t,
                    "services": services,
                    "call_edges": edges,
                    "label": {
                        "service": pod_to_service(fault["inject_pod"]),
                        "pod": fault["inject_pod"],
                        "fault_type": fault["inject_type"],
                    },
                },
                "metrics": series,
                "logs": {"fault": log_stats(log_files, max(len(log_files), 1)), "normal": normal_logs},
                "traces": {"fault": trace_stats(trace_files), "normal": normal_traces},
            }
            with gzip.open(out / f"{case_id}.json.gz", "wt") as fh:
                json.dump(bundle, fh)
            n += 1
            print(f"  {case_id}: {bundle['case']['label']['service']} / {fault['inject_type']}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/processed"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    total = sum(process_day(args.raw, day, args.out) for day in SYSTEMS)
    print(f"wrote {total} cases to {args.out}")


if __name__ == "__main__":
    main()
