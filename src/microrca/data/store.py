"""Load processed incident bundles and manage stratified dev/test splits."""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microrca.contracts import IncidentCase

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT_DIR / "data" / "processed"
RCAEVAL_DIR = ROOT_DIR / "data" / "rcaeval_processed"
DEFAULT_SPLITS_DIR = ROOT_DIR / "data" / "splits"


@dataclass
class CaseBundle:
    """An incident plus the telemetry the tools are allowed to query."""

    case: IncidentCase
    metrics: dict[str, dict[str, list]]
    logs: dict[str, dict[str, Any]]
    traces: dict[str, dict[str, Any]]

    def blind(self) -> IncidentCase:
        """The case as a method sees it: no label."""
        return self.case.model_copy(update={"label": None})

    def blind_bundle(self) -> CaseBundle:
        """Return a shallow copy of the bundle whose case has label=None (prevents leakage)."""
        return CaseBundle(
            case=self.blind(),
            metrics=self.metrics,
            logs=self.logs,
            traces=self.traces,
        )


def load_case(path: Path) -> CaseBundle:
    with gzip.open(path, "rt") as fh:
        raw = json.load(fh)
    return CaseBundle(
        case=IncidentCase.model_validate(raw["case"]),
        metrics=raw["metrics"],
        logs=raw["logs"],
        traces=raw["traces"],
    )


def make_stratified_split(
    cases: list[CaseBundle], dev_ratio: float = 0.30, seed: int = 42
) -> dict[str, Any]:
    """Deterministically partition cases into dev and test sets stratified by (system, fault_type)."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for b in sorted(cases, key=lambda c: c.case.case_id):
        ft = b.case.label.fault_type.value if b.case.label else "unknown"
        strata[(b.case.system, ft)].append(b.case.case_id)

    dev_ids: list[str] = []
    test_ids: list[str] = []
    strata_summary: dict[str, dict[str, int]] = {}
    for (sys_name, ft), ids in sorted(strata.items()):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_dev = max(1, round(len(shuffled) * dev_ratio)) if len(shuffled) > 1 else 0
        dev_part = sorted(shuffled[:n_dev])
        test_part = sorted(shuffled[n_dev:])
        dev_ids.extend(dev_part)
        test_ids.extend(test_part)
        strata_summary[f"{sys_name}:{ft}"] = {
            "total": len(ids),
            "dev": len(dev_part),
            "test": len(test_part),
        }

    dev_ids.sort()
    test_ids.sort()
    return {
        "dataset": "nezha",
        "seed": seed,
        "dev_ratio": dev_ratio,
        "total_cases": len(cases),
        "dev_count": len(dev_ids),
        "test_count": len(test_ids),
        "strata": strata_summary,
        "dev": dev_ids,
        "test": test_ids,
    }


def write_splits(
    data_dir: Path = DEFAULT_DIR,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    dev_ratio: float = 0.30,
    seed: int = 42,
) -> dict[str, Any]:
    paths = sorted(Path(data_dir).glob("*.json.gz"))
    cases = [load_case(p) for p in paths]
    manifest = make_stratified_split(cases, dev_ratio=dev_ratio, seed=seed)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "nezha_splits.json").write_text(json.dumps(manifest, indent=2))
    (splits_dir / "dev.txt").write_text("\n".join(manifest["dev"]) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(manifest["test"]) + "\n")
    return manifest


def load_split_ids(split: str, splits_dir: Path = DEFAULT_SPLITS_DIR) -> set[str]:
    if split in ("all", None, ""):
        return set()
    splits_dir = Path(splits_dir)
    manifest_path = splits_dir / "nezha_splits.json"
    if not manifest_path.exists():
        write_splits(DEFAULT_DIR, splits_dir)
    manifest = json.loads(manifest_path.read_text())
    if split not in ("dev", "test"):
        raise ValueError(f"Unknown split {split!r}; expected 'all', 'dev', or 'test'")
    return set(manifest[split])


def load_cases(
    data_dir: Path = DEFAULT_DIR,
    system: str | None = None,
    limit: int | None = None,
    split: str | None = None,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
) -> list[CaseBundle]:
    paths = sorted(Path(data_dir).glob("*.json.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No processed cases in {data_dir}. Run scripts/fetch_nezha.sh and scripts/preprocess_nezha.py."
        )
    cases = [load_case(p) for p in paths]
    if split and split != "all":
        allowed = load_split_ids(split, splits_dir=splits_dir)
        # If evaluating on an external held-out set (e.g., RCAEval) where case_ids aren't in nezha_splits.json,
        # treat all external cases as held-out test cases when split == "test".
        matching = [c for c in cases if c.case.case_id in allowed]
        if matching or Path(data_dir).resolve() == DEFAULT_DIR.resolve():
            cases = matching
        elif split == "dev":
            cases = []
    if system:
        cases = [c for c in cases if c.case.system == system]
    return cases[:limit] if limit else cases
