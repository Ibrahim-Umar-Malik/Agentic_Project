"""Parse model output into the Diagnosis contract, tolerating code fences and prose."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from microrca.contracts import Diagnosis, FaultType, RankedCause


def _first_json_object(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start >= 0:
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


def parse_diagnosis(text: str, case_id: str, method: str) -> Diagnosis:
    obj = _first_json_object(text) or {}
    ranking = []
    for item in obj.get("ranking", []):
        try:
            ft = item.get("fault_type", "unknown")
            item["fault_type"] = ft if ft in FaultType._value2member_map_ else "unknown"
            item["confidence"] = min(max(float(item.get("confidence", 0.5)), 0.0), 1.0)
            ranking.append(RankedCause.model_validate(item))
        except (ValidationError, TypeError, ValueError):
            continue
    return Diagnosis(case_id=case_id, method=method, ranking=ranking, summary=str(obj.get("summary", ""))[:500])
