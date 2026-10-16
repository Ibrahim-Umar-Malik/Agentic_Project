"""Structured contracts shared by the data pipeline, tools, baselines and agents.

Every component exchanges these pydantic models (or their JSON form), so the
LLM output, the evaluation harness and the stored results all agree on one schema.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FaultType(str, Enum):
    CPU_CONTENTION = "cpu_contention"
    CPU_CONSUMED = "cpu_consumed"
    NETWORK_DELAY = "network_delay"
    EXCEPTION = "exception"
    RETURN = "return"  # code-level fault: wrong return value
    UNKNOWN = "unknown"


class GroundTruth(BaseModel):
    service: str
    pod: str
    fault_type: FaultType


class IncidentCase(BaseModel):
    """One labelled failure case, as served to every RCA method."""

    case_id: str
    system: Literal["online-boutique", "train-ticket"]
    alert_time: int = Field(description="Unix seconds when the anomaly is reported")
    services: list[str]
    call_edges: list[tuple[str, str]] = Field(default_factory=list, description="caller -> callee")
    label: GroundTruth | None = None  # hidden from methods during evaluation


class RankedCause(BaseModel):
    service: str
    fault_type: FaultType = FaultType.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: list[str] = Field(default_factory=list)


class Diagnosis(BaseModel):
    """The single output contract every RCA method must return."""

    case_id: str
    method: str
    ranking: list[RankedCause]
    summary: str = ""


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    ok: bool
    content: Any


class Usage(BaseModel):
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class AgentState(BaseModel):
    """Working state carried through the agent loop (the 'context' of the run)."""

    case: IncidentCase
    step: int = 0
    max_steps: int = 12
    messages: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[ToolResult] = Field(default_factory=list)
    hypotheses: list[RankedCause] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    done: bool = False


class EvalRecord(BaseModel):
    case_id: str
    system: str
    method: str
    true_service: str
    true_fault: str
    predicted: list[str]
    predicted_fault: str
    usage: Usage
