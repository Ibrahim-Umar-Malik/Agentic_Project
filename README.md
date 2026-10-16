# MicroRCA-Agent: Agentic Root Cause Analysis for Microservice Systems

AI4015 Agentic AI semester project, group **G2-07**: *Improving the Quality of Microservice-Based Systems Using Agentic AI*.

When a microservice system fails, the symptom (slow checkout, failed orders) usually shows up far from the cause. This project builds an LLM agent that investigates metrics, logs, traces and the service call graph with tools, the way an on-call engineer would, and returns a ranked, evidence-backed list of root-cause services and fault types. Phase 1 delivers the data pipeline, tool layer, provider setup, structured contracts, two non-agentic baselines with results, and an end-to-end agent loop.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                          # 6 smoke tests, no API key needed

# B0: statistical baselines (no LLM), real results on all 101 cases
python -m microrca.eval.run --method stat          # metrics only
python -m microrca.eval.run --method stat-mm       # metrics + logs + traces, rank fusion

# Offline end-to-end run of the LLM pipelines with the scripted stand-in model
python -m microrca.eval.run --method agent --provider scripted --out results/smoke

# Real LLM runs (pick any provider)
export OPENAI_API_KEY=...       # or ANTHROPIC_API_KEY / GEMINI_API_KEY
python -m microrca.eval.run --method single --provider openai --model gpt-4o-mini --limit 20
python -m microrca.eval.run --method agent  --provider openai --model gpt-4o-mini --limit 20
# Local / free: any OpenAI-compatible server, e.g. Ollama
OPENAI_BASE_URL=http://localhost:11434/v1 python -m microrca.eval.run --method agent --provider openai --model qwen2.5:14b
```

Each run writes `results/<method>.json` (AC@k, Avg@5, fault-type accuracy, LLM calls, tool calls, tokens, latency; overall, per system and per fault type) and `results/<method>_cases.jsonl` (one record per case).

## Data

We use the public multi-modal fault-injection dataset released with Nezha (Yu et al., ESEC/FSE 2023, MIT licence): **101 labelled failure cases** on two benchmark systems, each with metrics, logs, traces and a ground-truth root-cause service and fault type.

| System | Services | Cases | Fault types |
|---|---|---|---|
| Online Boutique | 10 | 56 | cpu_contention, cpu_consumed, network_delay, exception, return |
| Train-Ticket | 47 | 45 | cpu_contention, network_delay, exception, return |

The raw data is 3.2 GB, so the repository ships compact per-case bundles (`data/processed/*.json.gz`, 3.5 MB) built by `scripts/preprocess_nezha.py`. To rebuild them from scratch run `bash scripts/fetch_nezha.sh`. RCAEval (Pham et al., WWW 2025) RE2/RE3 cases are planned as a second, held-out evaluation set in Phase 2.

## Initial results (Phase 1)

Service-level root-cause localisation on all 101 cases:

| Method | Type | AC@1 | AC@3 | AC@5 | Avg@5 | OB AC@1 | TT AC@1 |
|---|---|---|---|---|---|---|---|
| B0-stat-metric | statistical, metrics only | 0.178 | 0.356 | 0.416 | 0.323 | 0.321 | 0.000 |
| B0-stat-multimodal | statistical, metrics + logs + traces (RRF) | **0.307** | **0.396** | **0.475** | **0.386** | 0.429 | 0.156 |
| B1-single-call | one LLM call over pre-computed summaries | run with your API key | | | | | |
| P0-react-agent | tool-using agent (proposed, skeleton) | run with your API key | | | | | |

By fault type, B0-stat-multimodal reaches AC@1 0.91 on CPU contention but only 0.10 on exceptions, 0.11 on wrong-return code faults and 0.03 on network delay. Code-level and propagating faults are where static scoring fails, and they are the target of the agent. Results from the `scripted` provider in `results/smoke/` only prove the pipeline works and must not be reported as LLM results.

## Repository layout

```
src/microrca/
  contracts.py            pydantic contracts: IncidentCase, Diagnosis, RankedCause, AgentState, Usage, EvalRecord
  data/store.py           loads processed incident bundles; blind() hides the label from methods
  tools/telemetry.py      6 observability tools + JSON-schema specs for function calling
  llm/providers.py        OpenAI / OpenAI-compatible / Gemini / Anthropic over httpx, plus offline 'scripted'
  baselines/statistical.py  B0: robust z-score and multimodal rank fusion
  baselines/single_call.py  B1: one LLM call, no tools, no iteration
  agent/react_agent.py    P0: ReAct tool loop with budgets, usage accounting, memory and verifier hooks
  agent/parsing.py        tolerant JSON -> Diagnosis parsing
  eval/metrics.py         AC@k, Avg@k, fault-type accuracy, cost/latency summaries
  eval/run.py             CLI evaluation runner
scripts/                  fetch_nezha.sh, preprocess_nezha.py
data/processed/           101 compact case bundles
results/                  baseline results; results/smoke for offline pipeline checks
docs/architecture.md      architecture and where each research gap is addressed
tests/                    smoke tests
```

## Agent tools

| Tool | Returns |
|---|---|
| `list_services` | services of the system |
| `get_call_graph(service?)` | caller/callee edges |
| `detect_metric_anomalies(top_k)` | (service, metric) pairs ranked by robust z-score after the alert |
| `get_metric_summary(service, metric)` | before mean/std and after values for one metric |
| `get_log_summary(service)` | log and error-line rates vs. fault-free, new log templates |
| `get_trace_summary(service)` | span rate and p50/p90 latency vs. fault-free |

## Team

| Member | Phase 1 responsibility |
|---|---|
| Member 1 (group head) | literature review, base paper analysis, proposal |
| Member 2 | data pipeline, tools, evaluation harness |
| Member 3 | LLM providers, baselines, agent loop |

Every member should commit their own part (see the commit history requirement in the Phase 1 brief).
