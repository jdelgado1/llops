# Fabric ↔ Foundry Integration Guide

**Owner (Foundry/LLMOps):** Jose Delgado  
**Owner (Fabric/Data):** Blu Gotlieb  
**Last updated:** 2026-06-22

---

## Overview

This document specifies the data contract between Foundry (traces + evals) and Fabric (ingestion + scorecards + exports). 

**TL;DR:** Jose exports to Fabric paths; Blu ingests, computes metrics, and exports curated datasets back to Foundry.

---

## Part 1: Foundry → Fabric (Jose's responsibility)

### 1.1 Foundry Traces → Fabric

**Location:** `Files/llmops/raw/foundry_traces/`

**Format:** JSONL (one record per line)

**Each record (agent call trace):**
```json
{
  "timestamp": "2026-06-22T14:30:00Z",
  "model": "qwen3-32b.ft-...",
  "agent": "tmg_ops_agent",
  "request": "Open a high priority support ticket for customer CU-9012 about a billing error.",
  "predicted_tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "create_support_ticket",
        "arguments": "{\"customer_id\": \"CU-9012\", \"issue_type\": \"billing\", \"priority\": \"high\"}"
      }
    }
  ],
  "latency_ms": 142,
  "tokens_used": 89
}
```

**When:** After each production model inference via Foundry Hosted Agent or post-deployment testing.

**Volume:** ~1–10k records per week (production traffic).

**How:** Use `src/llmops/fabric_integration.py::FabricExporter.export_traces()` or direct Foundry Tracing API dump.

---

### 1.2 Foundry Eval Results → Fabric

**Location:** `Files/llmops/raw/foundry_evals/<eval_run_name>/`

**Example:** `Files/llmops/raw/foundry_evals/baseline-vs-student-20260622/`

**Contents:**

#### **a) Summary (JSON)**
```json
{
  "timestamp": "2026-06-22T15:45:00Z",
  "eval_pool_size": 114,
  "eval_run_name": "baseline-vs-student-20260622",
  "models": {
    "baseline": { "ast_accuracy": 55.3 },
    "student": { "ast_accuracy": 88.6 },
    "teacher": { "ast_accuracy": 91.2 }
  }
}
```

**File:** `eval_results_<timestamp>.json`

#### **b) Per-Item Details (JSONL)**
```json
{
  "eval_item_id": 0,
  "request": "Open a high priority support ticket for customer CU-9012...",
  "reference_tool_calls": [{"function": {"name": "create_support_ticket", "arguments": {...}}}],
  "predicted_by_model": {
    "baseline": [...],
    "student": [...],
    "teacher": [...]
  },
  "ast_match_by_model": {
    "baseline": false,
    "student": true,
    "teacher": true
  }
}
```

**File:** `eval_details_<timestamp>.jsonl`

**When:** After each eval run (typically after training a new student model).

**How:** Use `scripts/eval_tool_with_fabric_export.py --fabric-export`.

---

## Part 2: Fabric → Foundry (Blu's responsibility)

### 2.1 Fabric Exports for Foundry

**Location:** `Files/llmops/foundry_exports/<dataset_version>/`

**Example:** `Files/llmops/foundry_exports/golden-drift-corrected-20260622/`

**Format:** JSONL (Foundry SFT format)

**Each record (retraining candidate):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a tool-calling assistant..."},
    {"role": "user", "content": "Open a ticket for CU-9012 about billing."},
    {"role": "assistant", "tool_calls": [{"function": {"name": "create_support_ticket", "arguments": "{...}"}}]}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "create_support_ticket",
        "description": "...",
        "parameters": {...}
      }
    }
  ]
}
```

**Purpose:** Approved golden traces (from scorecards) ready for Foundry retraining.

**When:** After Blu runs drift detection, retraining candidate generation, and approval gates.

**How (for Foundry to consume):**
- **Option A (SDK):** Python `FabricExporter.read_export_dataset(dataset_version)` → upload to Foundry via Python SDK
- **Option B (Manual):** Download JSONL, upload to Foundry Fine-tune UI
- **Option C (Direct):** Can Foundry SDK directly register paths on Fabric storage? (Needs confirmation from Blu + Foundry docs)

---

## Part 3: Blu's Ingestion & Processing

Blu's notebooks in `lh_llmops` handle:

1. **Ingest traces** (`Files/llmops/raw/foundry_traces/`) → Table: `foundry_traces`
2. **Ingest eval results** (`Files/llmops/raw/foundry_evals/`) → Tables: `eval_runs`, `eval_details`
3. **Compute scorecards** → Metrics: AST accuracy by model over time
4. **Detect drift** → Alert if accuracy drops > threshold
5. **Generate retraining candidates** → Filter traces by quality, recency, error patterns
6. **Export approved dataset** → Write JSONL to `Files/llmops/foundry_exports/<dataset_version>/`
7. **Dashboard** → Power BI on scorecards + drift trends

---

## Part 4: Integration Checklist

### Jose (Foundry side) — Immediate

- [ ] **Create eval pool JSONL** for held-out test set (114 items, no training leakage)
  - Location: `artifacts/eval_pool_114items.jsonl`
  - Format: Same as training SFT, with `reference_tool_calls` added
  
- [ ] **Run eval script:**
  ```bash
  python scripts/eval_tool_with_fabric_export.py \
    --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
    --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
    --student-deployment <STUDENT_FINETUNED_DEPLOYMENT> \
    --eval-run-name baseline-vs-student-20260622 \
    --fabric-lakehouse-root /Volumes/lh_llmops \
    --fabric-export
  ```

- [ ] **Confirm paths readable** by Fabric (requires Databricks volume mount + Foundry integration)

### Blu (Fabric side) — Once Jose exports

- [ ] **Ingest traces** from `Files/llmops/raw/foundry_traces/`
  - Parse JSONL, denormalize into `foundry_traces` table
  - Schema: timestamp, model, agent, request, tool_calls, latency_ms, tokens_used

- [ ] **Ingest eval results** from `Files/llmops/raw/foundry_evals/`
  - Summary JSON → `eval_runs` table
  - Details JSONL → `eval_details` table
  - Schema: eval_id, eval_run_name, item_id, model, reference_calls, predicted_calls, ast_match

- [ ] **Scorecard notebook**
  - Compute AST accuracy trends (baseline vs student vs teacher)
  - Plot: accuracy % over time
  - Highlight: student ≥ teacher (success) or student < baseline (investigate)

- [ ] **Drift detection notebook**
  - Monitor recent traces for accuracy drops
  - Trigger if accuracy < threshold (e.g., < baseline * 0.9)
  - Log drift event with date/reason

- [ ] **Retraining candidate generation**
  - Filter `eval_details` for ast_match=false (errors)
  - OR sample recent high-confidence traces
  - Convert to Foundry SFT format (messages + tools + tool_calls)
  - Export to `Files/llmops/foundry_exports/golden-drift-corrected-<date>/`

- [ ] **Confirm Foundry can consume** from Fabric exports
  - If direct path registration works: great, document it
  - If not: Jose will pull JSONL via SDK and upload

---

## Part 5: Questions for Blu & Foundry

1. **Path access:** Can Foundry/Python SDK read from Databricks volume mount (`/Volumes/lh_llmops/...`)?
   - If yes: Jose can export traces directly via Fabric SDK (faster, no intermediate storage)
   - If no: Jose exports to Azure Blob, Blu ingests via SAS URL

2. **Foundry SDK consumption:** Can `azure-ai` or Foundry Python SDK directly register JSONL from Fabric paths, or must we download then upload?
   - Affects: Automation of retraining loop

3. **Eventthouse:** Are Blu's tables in Eventthouse or SQL? (Affects query patterns + connector choices)

---

## Part 6: File Manifest

### Jose's code (Foundry side):

- `src/llmops/fabric_integration.py` — Exporter class + helper functions
- `scripts/eval_tool_with_fabric_export.py` — 3-way eval runner with Fabric export
- `artifacts/eval_pool_114items.jsonl` — Held-out eval pool (to create)

### Blu's code (Fabric side):

- `notebooks/01_ingest_foundry_traces.py` — Load traces, populate `foundry_traces` table
- `notebooks/02_ingest_foundry_evals.py` — Load evals, populate `eval_runs` + `eval_details` tables
- `notebooks/03_scorecard.py` — Compute AST accuracy trends, create Power BI dataset
- `notebooks/04_drift_detection.py` — Monitor for accuracy drops, log alerts
- `notebooks/05_retraining_candidates.py` — Filter errors, export curated dataset to Foundry
- `dashboards/aml_scorecard.pbix` — Power BI dashboard on accuracy + drift

---

## Next Steps (in order)

1. **Jose:** Create eval_pool_114items.jsonl (from original 300, held-out 20%)
2. **Jose:** Run eval script with `--fabric-export` (or `--fabric-debug` first to test locally)
3. **Blu:** Ingest traces + evals into Fabric tables
4. **Blu:** Run scorecard → confirm baseline < student < teacher (or investigate if not)
5. **Blu:** Run drift detection (currently no drift; baseline for future)
6. **Blu:** Run retraining candidate generation → export first batch to Foundry
7. **Jose:** Confirm Foundry can consume from `Files/llmops/foundry_exports/`
8. **Loop:** Next iteration starts from step 1 (re-run eval, export, ingest, retrain)

---

## Quick Reference: CLI Commands

**Jose (Foundry eval + export):**
```bash
# Run eval with Fabric export (debug mode first)
python scripts/eval_tool_with_fabric_export.py \
  --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
  --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
  --student-deployment <from Foundry> \
  --eval-run-name baseline-vs-student-20260622 \
  --fabric-debug  # local export to ./fabric_export_debug/

# Real export (requires Fabric connectivity)
python scripts/eval_tool_with_fabric_export.py \
  --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
  --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
  --student-deployment <from Foundry> \
  --eval-run-name baseline-vs-student-20260622 \
  --fabric-lakehouse-root /Volumes/lh_llmops \
  --fabric-export
```

**Blu (Fabric side — Databricks):**
```sql
-- Check ingested eval results
SELECT eval_run_name, model, ast_accuracy FROM eval_runs ORDER BY timestamp DESC LIMIT 10;

-- Check per-item details
SELECT item_id, ast_match_by_model FROM eval_details WHERE ast_match_by_model LIKE '%false%';

-- List retraining candidates
SELECT COUNT(*) as error_count FROM eval_details WHERE ast_match_by_model['student'] = false;
```

---

## Contact & Notes

- **Jose:** Foundry/eval/distillation loop. Questions about traces, eval format, model endpoints → Jose
- **Blu:** Fabric ingestion, scorecards, drift, retraining. Questions about table schemas, Power BI connectors → Blu
- **Ambiguities:** See Part 5 (Questions). Resolve async and update this doc.

---

*This is a living document. Update as integration details become clear.*
