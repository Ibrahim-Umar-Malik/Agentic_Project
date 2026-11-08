"""Smoke and Phase 2 unit tests: data splits, contracts, tools, budget, memory leakage, verifier, and metrics."""
from pathlib import Path
import pytest

from microrca.agent import react_agent
from microrca.agent.parsing import parse_diagnosis
from microrca.baselines import openrca, single_call, statistical
from microrca.contracts import AgentState, EvalRecord, Usage
from microrca.data.store import (
    DEFAULT_DIR,
    DEFAULT_SPLITS_DIR,
    RCAEVAL_DIR,
    load_cases,
    load_split_ids,
    make_stratified_split,
)
from microrca.eval.metrics import ac_at_k, aggregate_seed_summaries, summarize
from microrca.gaps.budget import BudgetConfig, BudgetController
from microrca.gaps.memory import IncidentMemory
from microrca.gaps.verifier import CounterfactualVerifier
from microrca.llm.providers import Scripted
from microrca.tools.telemetry import TOOLS, call_tool

CASES = load_cases(limit=3)


def test_cases_have_labels_and_telemetry():
    c = CASES[0]
    assert c.case.label is not None and c.case.label.service in c.case.services
    assert c.blind().label is None
    assert c.blind_bundle().case.label is None
    assert set(c.metrics) == set(c.case.services)


def test_every_tool_runs():
    b = CASES[0]
    svc = b.case.services[0]
    for name in TOOLS:
        args = {"service": svc} if name.startswith("get_") and name != "get_call_graph" else {}
        assert "error" not in call_tool(b, name, args), name
    assert "error" in call_tool(b, "nope", {})


def test_statistical_baseline_ranks_all_services():
    d = statistical.diagnose(CASES[0])
    assert len(d.ranking) == len(CASES[0].case.services)


def test_single_call_and_agent_run_offline(tmp_path: Path):
    llm = Scripted()
    d1, _ = single_call.diagnose(CASES[0], llm)
    d2, state = react_agent.run(
        CASES[0], llm, seed=42, run_id="pytest_run", log_dir=tmp_path
    )
    assert d1.ranking and d2.ranking
    assert state.usage.tool_calls >= 1 and state.done
    traj_file = tmp_path / "pytest_run" / f"{CASES[0].case.case_id}.jsonl"
    assert traj_file.exists()


def test_parse_diagnosis_tolerates_fences():
    d = parse_diagnosis('```json\n{"ranking": [{"service": "a", "fault_type": "weird"}]}\n```', "c", "m")
    assert d.ranking[0].service == "a" and d.ranking[0].fault_type.value == "unknown"


def test_ac_at_k_and_multi_seed_aggregation():
    r1 = EvalRecord(
        case_id="c", system="s", method="m", true_service="b", true_fault="x",
        predicted=["a", "b"], predicted_fault="x", usage=Usage(),
    )
    r2 = EvalRecord(
        case_id="c", system="s", method="m", true_service="a", true_fault="x",
        predicted=["a", "b"], predicted_fault="x", usage=Usage(),
    )
    assert ac_at_k([r1], 1) == 0 and ac_at_k([r1], 2) == 1
    agg = aggregate_seed_summaries([summarize([r1]), summarize([r2])])
    assert agg["mean"]["AC@1"] == 0.5
    assert "±" in agg["formatted"]["AC@1"]


def test_stratified_splits_are_disjoint_and_deterministic():
    all_cases = load_cases(DEFAULT_DIR)
    m1 = make_stratified_split(all_cases, dev_ratio=0.30, seed=42)
    m2 = make_stratified_split(all_cases, dev_ratio=0.30, seed=42)
    assert m1["dev"] == m2["dev"] and m1["test"] == m2["test"]
    dev_ids = load_split_ids("dev", DEFAULT_SPLITS_DIR)
    test_ids = load_split_ids("test", DEFAULT_SPLITS_DIR)
    assert len(dev_ids) == 30 and len(test_ids) == 71
    assert dev_ids.isdisjoint(test_ids)
    assert len(dev_ids | test_ids) == 101


def test_rcaeval_heldout_loads():
    if RCAEVAL_DIR.exists():
        rc_cases = load_cases(RCAEVAL_DIR, split="test")
        assert len(rc_cases) >= 10
        assert rc_cases[0].case.dataset.startswith("rcaeval")


def test_budget_controller_caching_and_early_stop():
    ctrl = BudgetController(BudgetConfig(max_steps=5, max_tool_calls=6, max_calls_per_step=2))
    state = AgentState(case=CASES[0].blind())
    calls = [
        {"id": "1", "name": "detect_metric_anomalies", "arguments": {"top_k": 5}},
        {"id": "2", "name": "get_call_graph", "arguments": {}},
        {"id": "3", "name": "list_services", "arguments": {}},
    ]
    trimmed = ctrl.filter_tool_calls(calls, state)
    assert len(trimmed) == 2

    res1, cached1 = ctrl.execute_or_cache(call_tool, CASES[0], "list_services", {}, state)
    res2, cached2 = ctrl.execute_or_cache(call_tool, CASES[0], "list_services", {}, state)
    assert not cached1 and cached2 and res1 == res2 and ctrl.cache_hits == 1


def test_incident_memory_strictly_rejects_test_leakage():
    dev_cases = load_cases(DEFAULT_DIR, split="dev", limit=5)
    test_cases = load_cases(DEFAULT_DIR, split="test", limit=2)
    mem = IncidentMemory.from_dev_cases(dev_cases)
    assert mem.assert_no_test_leakage()

    # Attempting to index a test case MUST raise ValueError
    with pytest.raises(ValueError, match="DATA LEAKAGE VIOLATION"):
        mem.index_dev_cases(test_cases)

    # Storing a test run during evaluation must never persist labels
    llm = Scripted()
    diag, _ = react_agent.run(
        test_cases[0],
        llm,
        memory=mem,
        verifier=CounterfactualVerifier(),
        budget_controller=BudgetController(),
    )
    assert mem.assert_no_test_leakage()
    assert len(mem.unlabeled_runtime_traces) == 1
    assert "label" not in mem.unlabeled_runtime_traces[0]


def test_openrca_baseline_runs():
    llm = Scripted()
    diag, state = openrca.diagnose(CASES[0], llm)
    assert diag.method.startswith("E2-openrca:")
    assert diag.ranking and state.done
