"""Load processed incident bundles (see scripts/preprocess_nezha.py)."""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microrca.contracts import IncidentCase

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


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


def load_case(path: Path) -> CaseBundle:
    with gzip.open(path, "rt") as fh:
        raw = json.load(fh)
    return CaseBundle(
        case=IncidentCase.model_validate(raw["case"]),
        metrics=raw["metrics"],
        logs=raw["logs"],
        traces=raw["traces"],
    )


def load_cases(
    data_dir: Path = DEFAULT_DIR, system: str | None = None, limit: int | None = None
) -> list[CaseBundle]:
    paths = sorted(Path(data_dir).glob("*.json.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No processed cases in {data_dir}. Run scripts/fetch_nezha.sh and scripts/preprocess_nezha.py."
        )
    cases = [load_case(p) for p in paths]
    if system:
        cases = [c for c in cases if c.case.system == system]
    return cases[:limit] if limit else cases
