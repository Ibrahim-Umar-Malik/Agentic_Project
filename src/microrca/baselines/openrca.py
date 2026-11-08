"""Base-paper reproduction: OpenRCA-style iterative telemetry query agent (Xu et al., ICLR 2025).

OpenRCA formulates root-cause localization as an unverified multi-step agent loop over
metrics, logs, and traces with a fixed long step horizon (10 steps, 20 tool calls) and
no counterfactual call-graph verifier (Gap 1), no adaptive budget controller (Gap 2),
and no episodic incident memory (Gap 3).
"""
from __future__ import annotations

from pathlib import Path

from microrca.agent import react_agent
from microrca.contracts import AgentState, Diagnosis
from microrca.data.store import CaseBundle
from microrca.llm.providers import ChatModel


def diagnose(
    bundle: CaseBundle,
    llm: ChatModel,
    seed: int = 42,
    run_id: str = "",
    log_dir: Path | None = None,
) -> tuple[Diagnosis, AgentState]:
    blind = bundle.blind_bundle()
    return react_agent.run(
        blind,
        llm,
        max_steps=10,
        max_tool_calls=20,
        memory=None,
        verifier=None,
        budget_controller=None,
        seed=seed,
        run_id=run_id,
        log_dir=log_dir,
        method_tag=f"E2-openrca:{llm.name}",
    )
