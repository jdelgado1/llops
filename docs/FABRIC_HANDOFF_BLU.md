# Fabric ↔ Foundry Quick Start for Blu

**From:** Jose (Foundry/LLMOps)  
**To:** Blu (Fabric/Data platform)  
**Date:** 2026-06-22

---

## What's Ready Now

✅ **[docs/FABRIC_INTEGRATION.md](../docs/FABRIC_INTEGRATION.md)** — Full integration spec (read this first)

✅ **Code for Jose:**
- `src/llmops/fabric_integration.py` — Export traces/evals to Fabric paths
- `scripts/eval_tool_with_fabric_export.py` — Run 3-way eval + export
- `scripts/gen_eval_pool.py` — Generate held-out eval pool (no training leakage)

✅ **Eval pool ready:** `artifacts/eval_pool_114items.jsonl` (64 items, Foundry SFT format with reference_tool_calls)

---

## What Happens Next (Your Job)

### **Step 1: I (Jose) Run the Eval**
```bash
python scripts/eval_tool_with_fabric_export.py \
  --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
  --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
  --student-deployment <STUDENT_FINETUNED_DEPLOYMENT> \
  --eval-run-name baseline-vs-student-20260622 \
  --fabric-lakehouse-root /Volumes/lh_llmops \
  --fabric-export
```

This writes:
- **Traces** → `Files/llmops/raw/foundry_traces/`
- **Eval summary** → `Files/llmops/raw/foundry_evals/baseline-vs-student-20260622/eval_results_*.json`
- **Eval details** → `Files/llmops/raw/foundry_evals/baseline-vs-student-20260622/eval_details_*.jsonl`

### **Step 2: You (Blu) Ingest into Fabric**

Expected file format:

**eval_results_*.json:**
```json
{
  "timestamp": "2026-06-22T15:45:00Z",
  "eval_pool_size": 64,
  "eval_run_name": "baseline-vs-student-20260622",
  "models": {
    "baseline": { "ast_accuracy": 55.3 },
    "student": { "ast_accuracy": 88.6 },
    "teacher": { "ast_accuracy": 91.2 }
  }
}
```

**eval_details_*.jsonl** (one per line):
```json
{
  "eval_item_id": 0,
  "request": "Open a high priority support ticket...",
  "reference_tool_calls": [...],
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

**Your notebooks should:**

1. **Ingest summary** → Create `eval_runs` table:
   ```sql
   CREATE TABLE eval_runs (
     eval_run_name STRING,
     timestamp TIMESTAMP,
     eval_pool_size INT,
     model STRING,
     ast_accuracy DOUBLE
   )
   ```

2. **Ingest details** → Create `eval_details` table:
   ```sql
   CREATE TABLE eval_details (
     eval_run_name STRING,
     eval_item_id INT,
     request STRING,
     reference_tool_calls JSON,
     predicted_by_model MAP<STRING, ARRAY<JSON>>,
     ast_match_by_model MAP<STRING, BOOLEAN>
   )
   ```

3. **Compute scorecard** → Query for trending:
   ```sql
   SELECT 
     eval_run_name,
     model,
     ast_accuracy,
     timestamp
   FROM eval_runs
   ORDER BY timestamp DESC
   LIMIT 100
   ```

4. **Detect drift** → Alert if `ast_accuracy < 50%` or drops > 10% from previous run

5. **Generate retraining candidates** → Filter `eval_details` for `ast_match_by_model['student'] = false` → export to `Files/llmops/foundry_exports/`

---

## Fabric Paths (Your Directories)

```
lh_llmops/
├── Files/llmops/
│   ├── raw/
│   │   ├── foundry_traces/              ← I export here (agent call logs)
│   │   └── foundry_evals/
│   │       └── baseline-vs-student-20260622/
│   │           ├── eval_results_*.json  ← Summary (model accuracies)
│   │           └── eval_details_*.jsonl ← Per-item details
│   ├── foundry_exports/
│   │   └── golden-drift-corrected-20260622/
│   │       └── data.jsonl               ← You export here (retraining data)
│   └── (other paths for your Fabric notebooks)
```

---

## Key Questions to Resolve

1. **Path connectivity:** Can your Fabric notebook read from `/Volumes/lh_llmops/Files/...` without issues?
   - If yes: Great, you can ingest directly
   - If no: Let me know; I can export to Azure Blob instead and you pull via SAS

2. **Foundry SDK consumption:** Once you export curated data to `foundry_exports/`, can Foundry consume it directly?
   - If yes: I can automate re-training
   - If no: I'll download + re-upload via Foundry UI

3. **Table engine:** Are your tables in Eventthouse or Delta Lake? (Affects query patterns)

---

## Timeline

- **Now:** Jose runs eval + exports to Fabric (wait 2-5 min)
- **Then:** Blu ingests, computes scorecard
- **Demo day (6/26):** Show 3-way comparison + drift trends on Power BI
- **Loop:** Repeat eval → ingest → scorecard weekly

---

## Contact & Troubleshooting

- **Jose:** Questions about eval format, model endpoints, Foundry tracing
- **Blu:** Questions about Fabric ingestion, table schemas, drift logic
- **Together:** Resolve integration ambiguities (see FABRIC_INTEGRATION.md Part 5)

---

## Next: Run the Eval

When you're ready:
1. I'll run `eval_tool_with_fabric_export.py` with `--fabric-export`
2. Files land in your Fabric paths
3. You ping me once you've ingested + scorecard is live
4. We iterate

🚀 Ready to go?
