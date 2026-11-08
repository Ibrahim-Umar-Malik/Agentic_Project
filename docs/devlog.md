# MicroRCA-Agent Development Log (Group G2-07)

> **Authenticity Policy**: This log records only verified implementation milestones and actual offline/dev diagnostics executed in this repository. A structured template is provided at the bottom for logging future live-API LLM runs (`openai`, `gemini`, `anthropic`, or local `ollama`).

---

## Verified Implementation Milestones

### Entry 1 — Phase 1 Foundation
- **Scope**: Built `scripts/preprocess_nezha.py` converting 101 raw Nezha failure cases (56 Online Boutique, 45 Train-Ticket) into compact `data/processed/*.json.gz` bundles; implemented `contracts.py`, 6 observability tools in `tools/telemetry.py`, `B0-stat-metric`, `B0-stat-multimodal`, `B1-single-call`, and the `P0` ReAct agent skeleton.
- **Finding**: `B0-stat-multimodal` achieved `AC@1 = 0.307` across all 101 cases (`0.913` on `cpu_contention`, but `<= 0.11` on `exception`, `return`, and `network_delay`), motivating Phase 2 gap modules.

### Entry 2 — Stratified Dev/Test Split & Leakage Prevention (Task T1)
- **Action**: Implemented `make_stratified_split()` and `write_splits()` in `src/microrca/data/store.py` using `seed=42` and a `30/70` dev/test ratio stratified by `(system, fault_type)`.
- **Artifact**: `data/splits/nezha_splits.json` (`30` dev cases in `data/splits/dev.txt`, `71` test cases in `data/splits/test.txt`), plus 12 held-out RCAEval RE2/RE3 cases in `data/rcaeval_processed/`.
- **Verification**: Added `CaseBundle.blind_bundle()` and `IncidentMemory.assert_no_test_leakage()` to guarantee that test labels are never accessible to tools, prompts, or memory indexing.

### Entry 3 — Dev-Split Error Diagnosis & Gap 1-3 Implementation (Tasks T2, T3)
- **Diagnostic Observation on Dev Split (`split="dev"`, 30 cases)**:
  - Inspecting the 30 Dev cases revealed why naive log template counting failed on `exception` and `return` faults: chatty services (`recommendationservice`, `ts-cancel-service`, `ts-delivery-service`) emit `INFO` logs containing hex UUID substrings not stripped by basic number regexes, creating up to 40 benign "new templates" even when `error_lines_per_min == 0.0`.
  - Gating template anomalies on genuine error-rate increases (`err_delta > 0`) and propagating blame along `caller -> callee` edges (including caller error log hints such as `frontend` logging `"Failed to retrieve ads"` when `adservice` fails) raised offline Dev-split `AC@1` from `0.333` to `0.567` and `fault_type_acc|top1_hit` from `0.400` to `0.765`.
- **Modules Completed**:
  - `src/microrca/gaps/verifier.py` (`CounterfactualVerifier`)
  - `src/microrca/gaps/budget.py` (`BudgetController`)
  - `src/microrca/gaps/memory.py` (`IncidentMemory`)

### Entry 4 — Experiments E1–E6, Failure Analysis & Gap Table Pipeline (Tasks T4, T5, T6)
- **Action**: Created `experiments/run_e1_baselines.py` through `experiments/run_e6_efficiency.py`, `experiments/run_all.py`, `src/microrca/eval/failures.py`, and `src/microrca/eval/gap_table.py`.
- **Verification**: Executed the complete 3-seed (`42, 43, 44`) pipeline on the held-out Test split with `--provider scripted` (explicitly marked `"is_offline_scripted": true`), producing per-case trajectories in `logs/`, failure categories in `results/failure_categories.csv`, exported case studies in `results/case_studies/`, and `results/gap_resolution_table.md`.

---

## Template for Future Live LLM Runs (Do Not Fabricate)

Use the template below when recording runs with a live LLM provider (`openai`, `gemini`, `anthropic`, or local `ollama`):

```markdown
### Entry [N] — Live LLM Evaluation Run
- **Date / Commit**: YYYY-MM-DD / `<git-sha>`
- **Runner (Member)**: [Name]
- **Provider & Model**: `--provider <openai|gemini|anthropic> --model <model_name> --temperature 0.2`
- **Split & Seeds**: `--split test --seeds 42,43,44`
- **Command Executed**:
  `python experiments/run_all.py --provider <provider> --model <model> --seeds 42,43,44`
- **Summary Metrics (`mean ± std`)**:
  - `E2-OpenRCA`: AC@1 = `[ ]`, Avg@5 = `[ ]`, Fault-Type Acc = `[ ]`, Avg Tokens = `[ ]`
  - `E3-Proposed`: AC@1 = `[ ]`, Avg@5 = `[ ]`, Fault-Type Acc = `[ ]`, Avg Tokens = `[ ]`
- **Failures / Anomalies Observed**:
  - Unparseable outputs / HTTP retries: `[ ]`
  - Notable case studies inspected: `[ ]`
```
