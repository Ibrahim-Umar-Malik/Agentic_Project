# Phase 2 Gap-Resolution Comparison Table (Base Paper vs. Proposed MicroRCA-Agent)

> **Authenticity Note**: The numbers below were produced with `--provider scripted` (offline deterministic stand-in policy) to verify the end-to-end Phase 2 pipeline. Re-run `python experiments/run_all.py --provider openai --model gpt-4o-mini` (or `gemini`/`anthropic`) with a live API key in `.env` to populate live LLM numbers.

## 1. Per-Gap Resolution Table (Held-Out Test Split, 3 Seeds `mean ± std`)

| Research Gap | Target Metric(s) | Base Paper (`E2-OpenRCA`) | Isolated Gap (`P0 + Gap`) | Our Full System (`E3-Proposed`) | Resolution Summary |
|---|---|---|---|---|---|
| **Gap 1: Multi-Modal Counterfactual Verifier** (`gaps/verifier.py`) | `AC@1` / `Avg@5` / `Fault-Type Acc` | `0.113 ± 0.000` / `0.256 ± 0.000` / `0.250 ± 0.000` | `0.423 ± 0.000` / `0.555 ± 0.000` / `0.600 ± 0.000` | **`0.413 ± 0.007`** / **`0.556 ± 0.001`** / **`0.614 ± 0.010`** | De-noises UUID template bursts, discounts upstream caller latency symptoms, and verifies fault signatures. |
| **Gap 2: Adaptive Cost-Aware Budget Controller** (`gaps/budget.py`) | `Tool Calls` / `Tokens` / `Early-Stop Rate` | `9.610 ± 0.000` / `4826.1 ± 0.0` / `0.000 ± 0.000` | `6.000 ± 0.000` / `2986.3 ± 0.0` / `1.000 ± 0.000` | **`6.000 ± 0.000`** / **`3773.6 ± 0.0`** / **`1.000 ± 0.000`** | Caches duplicate tool queries, bounds step fan-out, and converges early once top suspects are covered. |
| **Gap 3: Leakage-Free Incident Memory** (`gaps/memory.py`) | `AC@1` / `AC@3` / `Avg@5` | `0.113 ± 0.000` / `0.296 ± 0.000` / `0.256 ± 0.000` | `0.127 ± 0.000` / `0.296 ± 0.000` / `0.265 ± 0.000` | **`0.413 ± 0.007`** / **`0.577 ± 0.000`** / **`0.556 ± 0.001`** | Retrieves Dev-only historical telemetry fingerprints via cosine similarity with zero test-set label leakage. |

## 2. Overall Localization & Efficiency Comparison (`E2-OpenRCA` vs. `E3-Proposed`)

| Metric | Base Paper (`E2-OpenRCA`) | Proposed Full System (`E3`) | Delta (`Our - Base`) |
|---|---|---|---|
| AC@1 | `0.113 ± 0.000` | **`0.413 ± 0.007`** | `+0.300` |
| AC@3 | `0.296 ± 0.000` | **`0.577 ± 0.000`** | `+0.281` |
| AC@5 | `0.324 ± 0.000` | **`0.648 ± 0.000`** | `+0.324` |
| Avg@5 | `0.256 ± 0.000` | **`0.556 ± 0.001`** | `+0.300` |
| Fault-Type Accuracy (Top-1 Hit) | `0.250 ± 0.000` | **`0.614 ± 0.010`** | `+0.364` |
| Avg LLM Calls | `3.990 ± 0.000` | **`3.000 ± 0.000`** | `-0.99` |
| Avg Tool Calls | `9.610 ± 0.000` | **`6.000 ± 0.000`** | `-3.61` |
| Avg Tokens | `4826.1 ± 0.0` | **`3773.6 ± 0.0`** | `-1052.5` |
| Avg Cost (USD) | `0.000874 ± 0.000000` | **`0.000688 ± 0.000000`** | `-0.000186` |
| Early-Stop Rate | `0.000 ± 0.000` | **`1.000 ± 0.000`** | `+1.000` |
