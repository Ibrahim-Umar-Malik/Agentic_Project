# Phase 2 Release Checklist — MicroRCA-Agent (Group G2-07)

> **Note**: Per project instructions, the repository is **not** git-tagged automatically. Review every item below before final submission.

## Task Verification Matrix (T0–T8)

- [x] **T0: Phase 1 Feedback & README Updates**
  - [x] Added `"Changes since Phase 1"` section to `README.md`.
  - [x] Addressed Phase 1 feedback: explicit dev/test isolation, OpenRCA base-paper reproduction (`E2`), concrete implementations of Gaps 1–3, full trajectory logging, and 3-seed evaluation.

- [x] **T1: Deterministic Stratified Data Splits & Held-Out RCAEval**
  - [x] Generated deterministic 30/70 stratified split (`seed=42`) of 101 Nezha cases in `data/splits/nezha_splits.json`, `data/splits/dev.txt` (30 cases), and `data/splits/test.txt` (71 cases).
  - [x] Added `--split {all,dev,test}` and `--dataset {nezha,rcaeval}` CLI flags to `src/microrca/eval/run.py`.
  - [x] Added `scripts/fetch_rcaeval.sh` and `scripts/preprocess_rcaeval.py` with compact held-out RE2/RE3 bundles in `data/rcaeval_processed/`.

- [x] **T2: ReAct Agent Pipeline & Trajectory Logging**
  - [x] Enforced `bundle.blind_bundle()` (`label=None`) throughout `src/microrca/agent/react_agent.py`, `src/microrca/baselines/single_call.py`, and `src/microrca/baselines/openrca.py`.
  - [x] Added HTTP retry/timeout handling and error tracking (`Usage.retries`, `AgentState.error`).
  - [x] Logged full step-by-step tool, message, and gap events to `logs/<run_id>/<case_id>.jsonl` with `logs/<run_id>/config.json`.

- [x] **T3: Toggleable Phase 2 Research-Gap Modules**
  - [x] **Gap 1 (`src/microrca/gaps/verifier.py`)**: Multi-modal counter-evidence & call-graph propagation verifier (`--verifier` / `--no-verifier`).
  - [x] **Gap 2 (`src/microrca/gaps/budget.py`)**: Adaptive cost-aware budget controller with tool-call caching, fan-out bounding, and convergence early stopping (`--budget` / `--no-budget`).
  - [x] **Gap 3 (`src/microrca/gaps/memory.py`)**: Leakage-free Dev-only incident memory with structural guard against test-set labels (`--memory` / `--no-memory`).
  - [x] Tuned hyperparameters strictly on the 30-case Dev split (`split="dev"`).

- [x] **T4: Experiments E1–E6 (3 Seeds, `mean ± std`)**
  - [x] `experiments/run_e1_baselines.py` (`B0-stat-metric`, `B0-stat-multimodal`, `B1-single-call`)
  - [x] `experiments/run_e2_openrca.py` (`E2-openrca` base paper reproduction)
  - [x] `experiments/run_e3_full_system.py` (`E3` proposed full system on Nezha Test + RCAEval held-out)
  - [x] `experiments/run_e4_gap_validation.py` (`P0` vs. `+Gap1`, `+Gap2`, `+Gap3`)
  - [x] `experiments/run_e5_ablation.py` (`Full` vs. `-Verifier`, `-Budget`, `-Memory`)
  - [x] `experiments/run_e6_efficiency.py` (cost, latency, tokens, tool calls, early-stop rate, retries)
  - [x] `experiments/run_all.py` master runner.

- [x] **T5: Failure Analysis & Case Studies**
  - [x] Implemented `src/microrca/eval/failures.py` outputting `results/failure_categories.csv` and `results/failure_details.json`.
  - [x] Exported 3 success case studies and 3 failure case studies (Markdown + JSONL trajectories) to `results/case_studies/`.

- [x] **T6: Auto-Generated Gap-Resolution Table**
  - [x] Implemented `src/microrca/eval/gap_table.py` generating `results/gap_resolution_table.md` (`Base paper result` vs `Our result` as `mean ± std`).

- [x] **T7: Repo Maintenance & Automated Tests**
  - [x] Updated `docs/architecture.md`, `docs/devlog.md`, `pyproject.toml`, `.env.example`, and `README.md`.
  - [x] Added comprehensive unit and integration tests in `tests/test_pipeline.py` covering splits, metrics, budget controller, verifier, and memory leakage prevention.

- [x] **T8: Integrity & Security Checks**
  - [x] No API keys in source code (`.env.example` provided; `.env` loaded via `load_dotenv()`).
  - [x] Offline `scripted` runs explicitly flagged (`"is_offline_scripted": true`).
  - [x] All `pytest -q` tests pass.
