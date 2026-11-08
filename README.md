# MicroRCA-Agent: Agentic Root Cause Analysis for Microservice Systems

AI4015 Agentic AI semester project, group **G2-07**: *Improving the Quality of Microservice-Based Systems Using Agentic AI*.

When a microservice system fails, the symptom (slow checkout, failed orders) usually shows up far from the cause. This project builds an LLM agent that investigates metrics, logs, traces and the service call graph with tools, the way an on-call engineer would, and returns a ranked, evidence-backed list of root-cause services and fault types.

---

## Changes since Phase 1

1. **Deterministic Stratified Dev/Test Splits & Held-Out RCAEval (T1)**:
   - Added [`make_stratified_split`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/data/store.py#L51-L91) and [`write_splits`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/data/store.py#L94-L107) creating a deterministic `30/70` stratified split (`seed=42`) of the 101 Nezha cases in `data/splits/` (`30` Dev cases in `data/splits/dev.txt`, `71` held-out Test cases in `data/splits/test.txt`).
   - Added `--split {all,dev,test}` and `--dataset {nezha,rcaeval}` flags to [`src/microrca/eval/run.py`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/eval/run.py).
   - Added [`scripts/fetch_rcaeval.sh`](file:///home/mustafa-ali/Agentic_Project-main/scripts/fetch_rcaeval.sh) and [`scripts/preprocess_rcaeval.py`](file:///home/mustafa-ali/Agentic_Project-main/scripts/preprocess_rcaeval.py) providing 12 held-out RCAEval (RE2/RE3) bundles in `data/rcaeval_processed/`.
2. **Complete ReAct Pipeline, Blind Execution & Trajectory Logging (T2)**:
   - Enforced [`CaseBundle.blind_bundle()`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/data/store.py#L28-L35) (`label=None`) before passing data to any agent, baseline, tool, or gap module.
   - Added retry/timeout resilience, `.env` credential loading ([`load_dotenv`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/llm/providers.py#L28-L41)), and full JSONL trajectory logging (`logs/<run_id>/<case_id>.jsonl` + `logs/<run_id>/config.json`) in [`src/microrca/agent/react_agent.py`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/agent/react_agent.py).
3. **Three Toggleable Research-Gap Modules Tuned Exclusively on Dev (T3)**:
   - **Gap 1 — Multi-Modal Counterfactual Verifier** ([`src/microrca/gaps/verifier.py`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/gaps/verifier.py), `--[no-]verifier`): Filters out benign hex/UUID template noise, cross-checks error surges against CPU/latency/trace slowdowns, and propagates blame from upstream callers to downstream callees along `caller -> callee` edges.
   - **Gap 2 — Adaptive Cost-Aware Budget Controller** ([`src/microrca/gaps/budget.py`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/gaps/budget.py), `--[no-]budget`): Deduplicates repeated tool queries via caching, bounds per-step tool fan-out, and triggers early stopping once top suspects have multi-modal coverage.
   - **Gap 3 — Leakage-Free Incident Memory** ([`src/microrca/gaps/memory.py`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/gaps/memory.py), `--[no-]memory`): Indexes telemetry fingerprints strictly from the 30-case Dev split (`ValueError` raised if any test case is indexed) and never stores ground-truth labels at runtime.
4. **Experiments E1–E6, Failure Analysis & Auto-Generated Gap Table (T4–T6)**:
   - Added experiment runners in `experiments/` (`run_e1_baselines.py`, `run_e2_openrca.py`, `run_e3_full_system.py`, `run_e4_gap_validation.py`, `run_e5_ablation.py`, `run_e6_efficiency.py`, `run_all.py`) supporting multi-seed (`42, 43, 44`) `mean ± std` evaluation.
   - Added [`src/microrca/eval/failures.py`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/eval/failures.py) (`results/failure_categories.csv` + exported case studies in `results/case_studies/`) and [`src/microrca/eval/gap_table.py`](file:///home/mustafa-ali/Agentic_Project-main/src/microrca/eval/gap_table.py) (`results/gap_resolution_table.md`).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                          # 12 unit & pipeline tests, no API key needed

# B0: statistical baselines (no LLM) on the 71-case held-out Test split
python -m microrca.eval.run --method stat --split test
python -m microrca.eval.run --method stat-mm --split test

# Run the proposed Phase 2 agent with toggleable gaps (offline scripted check)
python -m microrca.eval.run --method agent --all-gaps --provider scripted --split test --seeds 42,43,44

# Toggle individual gaps (--[no-]verifier, --[no-]budget, --[no-]memory)
python -m microrca.eval.run --method agent --verifier --no-budget --no-memory --provider scripted --split dev

# Run the full Phase 2 experiment suite (E1-E6 + Failure Analysis T5 + Gap Table T6)
python experiments/run_all.py --provider scripted --seeds 42,43,44

# Real LLM runs (copy .env.example to .env and set your API key)
cp .env.example .env
python experiments/run_all.py --provider openai --model gpt-4o-mini --seeds 42,43,44
python -m microrca.eval.run --method agent --all-gaps --provider gemini --model gemini-2.0-flash --split test --seeds 42,43,44
```

---

## Data & Splits

We use the public multi-modal fault-injection dataset released with Nezha (Yu et al., ESEC/FSE 2023, MIT licence): **101 labelled failure cases** on two benchmark systems, partitioned deterministically (`seed=42`, stratified by `system` and `fault_type`) into **30 Dev cases** (`data/splits/dev.txt`) and **71 held-out Test cases** (`data/splits/test.txt`), plus **12 held-out RCAEval RE2/RE3 cases** (`data/rcaeval_processed/`).

| Dataset / Split | System(s) | Services | Cases | Purpose |
|---|---|---|---|---|
| Nezha Dev (`--split dev`) | Online Boutique (17), Train-Ticket (13) | 10 / 47 | 30 | Hyperparameter tuning & IncidentMemory bank |
| Nezha Test (`--split test`) | Online Boutique (39), Train-Ticket (32) | 10 / 47 | 71 | Strictly held-out primary evaluation |
| RCAEval RE2/RE3 (`--dataset rcaeval`) | Online Boutique (6), Train-Ticket (6) | 10 / 47 | 12 | Second held-out generalization set |

---

## Phase 1 & Phase 2 Results Summary

### 1. All-101 Nezha Statistical Baselines (Phase 1 Reference)

| Method | Type | AC@1 | AC@3 | AC@5 | Avg@5 | OB AC@1 | TT AC@1 |
|---|---|---|---|---|---|---|---|
| B0-stat-metric | statistical, metrics only | 0.178 | 0.356 | 0.416 | 0.323 | 0.321 | 0.000 |
| B0-stat-multimodal | statistical, metrics + logs + traces (RRF) | **0.307** | **0.396** | **0.475** | **0.386** | 0.429 | 0.156 |

### 2. Phase 2 Held-Out Test Split (`--split test`, 71 Cases, 3 Seeds `mean ± std`)

> **Authenticity Note**: Rows marked `(scripted)` use the deterministic offline policy (`--provider scripted`) to verify the end-to-end tool loop and Phase 2 gap modules without an API key, and are flagged `"is_offline_scripted": true` in `results/`. Run `python experiments/run_all.py --provider <openai|gemini|anthropic>` to generate live LLM tables.

| Experiment / Method | Gaps Enabled | AC@1 | AC@3 | AC@5 | Avg@5 | Fault-Type Acc | Avg Tool Calls | Avg Tokens |
|---|---|---|---|---|---|---|---|---|
| `E1: B0-stat-metric` | None (stat) | `0.169` | `0.338` | `0.408` | `0.313` | `0.250` | `0.00` | `0.0` |
| `E1: B0-stat-multimodal` | None (stat RRF) | `0.282` | `0.380` | `0.465` | `0.369` | `0.550` | `0.00` | `0.0` |
| `E2: OpenRCA Base (scripted)` | None | `0.113 ± 0.000` | `0.296 ± 0.000` | `0.324 ± 0.000` | `0.256 ± 0.000` | `0.250 ± 0.000` | `9.61 ± 0.00` | `4826.1 ± 0.0` |
| `E4: P0 + Gap 1 (Verifier)` | Verifier | `0.423 ± 0.000` | `0.577 ± 0.000` | `0.648 ± 0.000` | `0.555 ± 0.000` | `0.600 ± 0.000` | `9.61 ± 0.00` | `4826.1 ± 0.0` |
| `E4: P0 + Gap 2 (Budget)` | Budget | `0.113 ± 0.000` | `0.296 ± 0.000` | `0.324 ± 0.000` | `0.256 ± 0.000` | `0.250 ± 0.000` | `6.00 ± 0.00` | `2986.3 ± 0.0` |
| `E4: P0 + Gap 3 (Memory)` | Memory | `0.127 ± 0.000` | `0.296 ± 0.000` | `0.324 ± 0.000` | `0.265 ± 0.000` | `0.222 ± 0.000` | `9.61 ± 0.00` | `5613.5 ± 0.0` |
| **`E3: Proposed Full System (scripted)`** | **Verifier + Budget + Memory** | **`0.413 ± 0.007`** | **`0.577 ± 0.000`** | **`0.648 ± 0.000`** | **`0.556 ± 0.001`** | **`0.614 ± 0.010`** | **`6.00 ± 0.00`** | **`3773.6 ± 0.0`** |

See [`results/gap_resolution_table.md`](file:///home/mustafa-ali/Agentic_Project-main/results/gap_resolution_table.md), [`results/failure_categories.csv`](file:///home/mustafa-ali/Agentic_Project-main/results/failure_categories.csv), and `results/case_studies/` for full breakdowns and trajectories.

---

## Repository layout

```
src/microrca/
  contracts.py              pydantic contracts: IncidentCase, Diagnosis, RankedCause, AgentState, Usage, EvalRecord
  data/store.py             loads processed bundles, manages stratified dev/test splits, blind_bundle()
  tools/telemetry.py        6 observability tools + JSON-schema specs for function calling
  llm/providers.py          OpenAI / Gemini / Anthropic over httpx (.env + retries + seeds), plus offline 'scripted'
  baselines/statistical.py  B0: robust z-score and multimodal rank fusion
  baselines/single_call.py  B1: one LLM call, no tools, no iteration
  baselines/openrca.py      E2: OpenRCA base-paper reproduction (unverified iterative telemetry loop)
  agent/react_agent.py      ReAct tool loop with toggleable Gaps 1-3 and JSONL trajectory logging
  agent/parsing.py          tolerant JSON -> Diagnosis parsing
  gaps/verifier.py          Gap 1: Multi-modal counter-evidence & call-graph propagation verifier
  gaps/budget.py            Gap 2: Adaptive cost-aware budget controller (caching + early stop)
  gaps/memory.py            Gap 3: Leakage-free Dev-only incident memory bank
  eval/metrics.py           AC@k, Avg@k, fault-type accuracy, multi-seed mean ± std aggregation
  eval/run.py               CLI evaluation runner (--split, --dataset, --seeds, --[no-]verifier/budget/memory)
  eval/failures.py          T5: Failure mode categorizer + case study trajectory exporter
  eval/gap_table.py         T6: Auto-generated Markdown table comparing Base Paper vs Proposed
experiments/                E1-E6 runners (run_e1_baselines.py .. run_e6_efficiency.py, run_all.py)
scripts/                    fetch_nezha.sh, preprocess_nezha.py, fetch_rcaeval.sh, preprocess_rcaeval.py
data/                       processed/ (101 Nezha), rcaeval_processed/ (12 RE2/RE3), splits/ (dev/test)
results/                    E1-E6 outputs, failure_categories.csv, gap_resolution_table.md, case_studies/
docs/                       architecture.md, devlog.md, release_checklist.md
tests/                      12 unit & integration tests (splits, metrics, budget, memory leakage, pipeline)
```

---

## Team

| Member | Roll No. | Responsibility |
|---|---|---|
| Ibrahim Umar Malik (group head) | 22I-2315 | literature review, base paper (OpenRCA) reproduction (`E2`), verifier (`Gap 1`), paper |
| Syed Mustafa Ali | 22I-2304 | data splits & RCAEval loader, tool layer, evaluation harness, budget controller (`Gap 2`) |
| Fasih Shahid Abbasi | 22I-2235 | LLM providers, agent loop & trajectory logger, incident memory (`Gap 3`), CLI |
