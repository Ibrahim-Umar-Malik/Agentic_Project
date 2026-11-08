# MicroRCA-Agent Phase 2 Architecture

![architecture](architecture.png)

The static diagram source is `architecture.svg`. Below is the complete Phase 2 dataflow and component diagram showing how the stratified splits, observability tools, ReAct Investigator, and three toggleable Research Gap modules interact:

```mermaid
flowchart TD
    subgraph DataLayer["1. Data & Stratified Split Layer (No Leakage)"]
        Nezha["Nezha Dataset (101 Cases)<br/>data/processed/*.json.gz"] --> Splits["Stratified Split (Seed=42, 30/70)<br/>data/splits/nezha_splits.json"]
        Splits --> DevSet["Dev Split (30 Cases)<br/>Used ONLY for tuning & Memory Bank"]
        Splits --> TestSet["Held-Out Test Split (71 Cases)<br/>Strictly blinded via blind_bundle()"]
        RCAEval["RCAEval RE2/RE3 Held-Out Set (12 Cases)<br/>data/rcaeval_processed/*.json.gz"] --> TestSet
    end

    subgraph AgentCore["2. ReAct Investigator & Tool Layer"]
        TestSet --> Blind["CaseBundle.blind_bundle()<br/>(Strips GroundTruth label)"]
        Blind --> Loop["ReAct Agent Loop<br/>src/microrca/agent/react_agent.py"]
        Loop <--> Tools["Observability Tool Layer (6 Tools)<br/>src/microrca/tools/telemetry.py<br/>metrics, logs, traces, call_graph"]
    end

    subgraph Gaps["3. Toggleable Phase 2 Research-Gap Modules"]
        DevSet -->|"Index Dev Labels Only"| Gap3["Gap 3: IncidentMemory<br/>src/microrca/gaps/memory.py<br/>Cosine Fingerprint Recall + Tie-Break"]
        Gap3 <-->|"Few-Shot Prior & Rerank"| Loop
        Gap2["Gap 2: BudgetController<br/>src/microrca/gaps/budget.py<br/>Tool Cache + Fanout Cap + Early Stop"] <-->|"Governs Step/Tool Budget"| Loop
        Loop -->|"Candidate Diagnosis"| Gap1["Gap 1: CounterfactualVerifier<br/>src/microrca/gaps/verifier.py<br/>UUID De-noising + Call-Graph Propagation"]
    end

    subgraph EvalLayer["4. Evaluation, Trajectories & Failure Analysis"]
        Gap1 --> Diag["Final Verified Diagnosis JSON"]
        Diag --> Traj["Trajectory Logs<br/>logs/<run_id>/<case_id>.jsonl"]
        Diag --> Evals["Experiments E1-E6 & Failure Analysis<br/>src/microrca/eval/"]
    end
```

## Component Status (Phase 2)

| Component | Code | Owner | Status |
|---|---|---|---|
| Stratified Dev/Test Splits & RCAEval RE2/RE3 | `src/microrca/data/store.py`, `scripts/preprocess_rcaeval.py`, `data/splits/` | Syed Mustafa Ali | Done |
| Structured Contracts (Phase 2 extensions) | `src/microrca/contracts.py` | Syed Mustafa Ali | Done |
| Observability Tool Layer (6 tools + schemas) | `src/microrca/tools/telemetry.py` | Syed Mustafa Ali | Done |
| LLM Provider Layer (`.env`, retries, seeds) | `src/microrca/llm/providers.py` | Fasih Shahid Abbasi | Done |
| ReAct Investigator Agent + Trajectory Logger | `src/microrca/agent/react_agent.py` | Fasih Shahid Abbasi | Done |
| **Gap 1: Counterfactual Multi-Modal Verifier** | `src/microrca/gaps/verifier.py` | Ibrahim Umar Malik | Done (Dev-tuned) |
| **Gap 2: Adaptive Cost-Aware Budget Controller** | `src/microrca/gaps/budget.py` | Syed Mustafa Ali | Done (Dev-tuned) |
| **Gap 3: Leakage-Free Incident Memory** | `src/microrca/gaps/memory.py` | Fasih Shahid Abbasi | Done (Dev-only indexing) |
| Baselines (`B0`, `B1`, `E2-OpenRCA`) | `src/microrca/baselines/` | Ibrahim Umar Malik | Done |
| Experiments (`E1`–`E6`), Failure Analysis & Gap Table | `experiments/`, `src/microrca/eval/failures.py`, `src/microrca/eval/gap_table.py` | All Members | Done |
