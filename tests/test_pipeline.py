"""Smoke tests: data contracts, tools, every method end to end on a few cases (no API key needed)."""
from microrca.agent import react_agent
from microrca.agent.parsing import parse_diagnosis
from microrca.baselines import single_call, statistical
from microrca.data.store import load_cases
from microrca.eval.metrics import ac_at_k
from microrca.contracts import EvalRecord, Usage
from microrca.llm.providers import Scripted
from microrca.tools.telemetry import TOOLS, call_tool

CASES = load_cases(limit=3)


def test_cases_have_labels_and_telemetry():
    c = CASES[0]
    assert c.case.label is not None and c.case.label.service in c.case.services
    assert c.blind().label is None
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


def test_single_call_and_agent_run_offline():
    llm = Scripted()
    d1, _ = single_call.diagnose(CASES[0], llm)
    d2, state = react_agent.run(CASES[0], llm)
    assert d1.ranking and d2.ranking
    assert state.usage.tool_calls >= 1 and state.done


def test_parse_diagnosis_tolerates_fences():
    d = parse_diagnosis('```json\n{"ranking": [{"service": "a", "fault_type": "weird"}]}\n```', "c", "m")
    assert d.ranking[0].service == "a" and d.ranking[0].fault_type.value == "unknown"


def test_ac_at_k():
    r = EvalRecord(case_id="c", system="s", method="m", true_service="b", true_fault="x",
                   predicted=["a", "b"], predicted_fault="x", usage=Usage())
    assert ac_at_k([r], 1) == 0 and ac_at_k([r], 2) == 1
