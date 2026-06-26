# TMG LLMOps Demo — Parallel Workstreams (Jose & Blu)

**Created:** 2026-06-22  
**Demo:** 2026-06-26 (4 days)  
**Owner:** Jose (Foundry) + Blu (Fabric)

---

## Current Status

✅ **Jose's side:**
- Trained student model in Foundry (**canonical 200-row file succeeded**)
- Created eval pipeline + Fabric export code
- Generated 64-item held-out eval pool (no leakage)
- Ready to run 3-way eval (baseline vs student vs teacher)

✅ **Blu's side:**
- Built Fabric workspace `Fine Tune Demo`
- Created Lakehouse `lh_llmops` with backend tables
- Ready to ingest traces/evals + compute scorecards

🔄 **Integration:** Wire Foundry → Fabric paths (NEW)

---

## Parallel Workstreams (Start Now)

### **Jose's Checklist (Next 1–2 hours)**

- [ ] **Get student model endpoint from Foundry**
  - Training job ID: ?
  - Deployment name: `qwen3-32b.ft-<...>`
  - Endpoint URL: ?
  - Update `.env` → `STUDENT_FINETUNED_DEPLOYMENT=`

- [ ] **Confirm baseline deployment is live**
  - Baseline: `qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer`
  - Test: Can you invoke it via `invoke_model()`?

- [ ] **Test eval script locally (dry-run, no Fabric export yet)**
  ```bash
  python scripts/eval_tool_with_fabric_export.py \
    --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
    --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
    --student-deployment <STUDENT_FINETUNED_DEPLOYMENT> \
    --eval-run-name baseline-vs-student-20260622 \
    --limit 5  # Test on 5 items first
  ```

- [ ] **Run full eval + export to Fabric (if path is accessible)**
  ```bash
  # Test with local export first
  python scripts/eval_tool_with_fabric_export.py \
    --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
    --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
    --student-deployment <STUDENT_FINETUNED_DEPLOYMENT> \
    --eval-run-name baseline-vs-student-20260622 \
    --fabric-debug  # Write to ./fabric_export_debug/ locally
  ```

- [ ] **If Fabric path is accessible, run real export**
  ```bash
  python scripts/eval_tool_with_fabric_export.py \
    --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
    --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
    --student-deployment <STUDENT_FINETUNED_DEPLOYMENT> \
    --eval-run-name baseline-vs-student-20260622 \
    --fabric-lakehouse-root /Volumes/lh_llmops \
    --fabric-export
  ```

- [ ] **Verify files in Fabric paths** (or local if debug mode)
  - Check: `eval_results_*.json` is valid
  - Check: `eval_details_*.jsonl` has 64 lines
  - Ping Blu: "Files are ready to ingest"

---

### **Blu's Checklist (Parallel, Can Start Now)**

- [ ] **Set up Fabric notebook structure**
  - `01_ingest_foundry_evals.py` — read JSON + JSONL, populate tables
  - `02_scorecard.py` — compute AST accuracy trends
  - `03_drift_detection.py` — monitor for drops
  - `04_retraining_candidates.py` — filter errors, export to foundry_exports/

- [ ] **Create / verify Fabric tables**
  ```sql
  CREATE TABLE eval_runs (
    eval_run_name STRING,
    timestamp TIMESTAMP,
    eval_pool_size INT,
    model STRING,
    ast_accuracy DOUBLE
  );

  CREATE TABLE eval_details (
    eval_run_name STRING,
    eval_item_id INT,
    request STRING,
    reference_tool_calls JSON,
    predicted_by_model MAP<STRING, ARRAY<JSON>>,
    ast_match_by_model MAP<STRING, BOOLEAN>
  );
  ```

- [ ] **Verify path access** (while Jose is testing)
  ```python
  # In Fabric notebook
  import os
  path = "/Volumes/lh_llmops/Files/llmops/raw/foundry_evals"
  print(os.listdir(path))  # Should show eval run directories once Jose exports
  ```

- [ ] **Ingest eval results** (once Jose exports)
  - Read `eval_results_*.json` → populate `eval_runs`
  - Read `eval_details_*.jsonl` → populate `eval_details`
  - Verify row counts match (64 items expected)

- [ ] **Compute first scorecard**
  ```sql
  SELECT 
    model,
    ROUND(ast_accuracy, 2) as accuracy_pct,
    eval_pool_size
  FROM eval_runs
  WHERE eval_run_name = 'baseline-vs-student-20260622'
  ORDER BY ast_accuracy DESC;
  ```
  **Expected result:**
  | model | accuracy_pct | eval_pool_size |
  |-------|--------------|----------------|
  | teacher | ~91.0 | 64 |
  | student | ~85.0 | 64 | (or better!)
  | baseline | ~55.0 | 64 |

- [ ] **Create Power BI dataset** (optional for demo)
  - Datasource: Fabric `eval_runs` table
  - Viz: Line chart (model accuracy over time) → placeholder

---

## Integration Checkpoints

### **Checkpoint 1: Eval Outputs Ready** (Target: today, 1–2 hrs)
- Jose: Eval script runs, produces JSON + JSONL
- Blu: Can see files in Fabric (or Jose shows local export)
- Decision: Proceed to ingestion? Or debug path/format issues?

### **Checkpoint 2: Fabric Ingestion Live** (Target: today +2 hrs)
- Blu: Tables are populated with eval results
- Jose: Confirms JSON/JSONL structure matches spec
- Decision: Can we move to scorecard + drift detection?

### **Checkpoint 3: Scorecard + Dashboard Ready** (Target: tomorrow)
- Blu: Power BI (or Fabric) shows 3-way comparison
- Jose: Confirms metric interpretation (AST accuracy %)
- Decision: Ready for demo?

---

## Foundry ↔ Fabric Contract

**Foundry exports to Fabric:**
```
/Volumes/lh_llmops/
├── Files/llmops/raw/foundry_traces/
│   └── traces_tmg_ops_agent_qwen3_32b_<date>.jsonl  (optional, post-deployment)
└── Files/llmops/raw/foundry_evals/<eval_run_name>/
    ├── eval_results_<timestamp>.json
    └── eval_details_<timestamp>.jsonl
```

**Fabric exports to Foundry (future):**
```
Files/llmops/foundry_exports/<dataset_version>/
└── data.jsonl  (SFT format: messages + tools + tool_calls)
```

---

## Blockers & Unknowns

1. **Foundry student deployment endpoint** — Jose to get from Foundry UI
2. **Fabric path access** — Can Foundry scripts write to `/Volumes/lh_llmops/...`? (Databricks mount?)
   - If no: Use Azure Blob instead (SAS URL)
3. **Foundry SDK consumption of Fabric exports** — Can Foundry SDK read from Fabric JSONL?
   - If no: Jose will download + re-upload manually

**Action:** Jose & Blu sync async on Slack if blockers hit.

---

## Success Criteria (by 6/26 demo)

✅ **3-way eval complete:** baseline < student ≈ teacher (AST accuracy %)  
✅ **Results in Fabric:** Scorecard visible in Power BI  
✅ **No leakage:** Eval pool is 20% held-out (enforced by ID hash)  
✅ **Retraining loop ready:** (Foundry → Fabric → Foundry cycle documented)  
✅ **Drift story ready:** (Show how future drops would trigger retrain)  

---

## Commands (Copy-Paste Ready)

**Jose — Run eval:**
```bash
cd c:\Users\josedelgado\Coding\tmg-research-agent-llmops
$env:PYTHONPATH="src"
.venv\Scripts\python.exe scripts/eval_tool_with_fabric_export.py `
  --eval-pool-jsonl artifacts/eval_pool_114items.jsonl `
  --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer `
  --student-deployment <PASTE_FOUNDRY_ENDPOINT> `
  --eval-run-name baseline-vs-student-20260622 `
  --fabric-debug
```

**Blu — Check Fabric ingestion:**
```python
import os
import json
import pandas as pd

path = "/Volumes/lh_llmops/Files/llmops/raw/foundry_evals/baseline-vs-student-20260622"
for fname in os.listdir(path):
    if fname.endswith('.json'):
        with open(f"{path}/{fname}") as f:
            data = json.load(f)
            print("Eval results:", data)
    elif fname.endswith('.jsonl'):
        with open(f"{path}/{fname}") as f:
            lines = [json.loads(l) for l in f]
            print(f"Loaded {len(lines)} eval detail records")
```

---

## Files Ready

- ✅ `src/llmops/fabric_integration.py` — Exporter class
- ✅ `scripts/eval_tool_with_fabric_export.py` — Eval runner + Fabric export
- ✅ `scripts/gen_eval_pool.py` — Eval pool generator
- ✅ `artifacts/eval_pool_114items.jsonl` — Held-out eval data (64 items)
- ✅ `docs/FABRIC_INTEGRATION.md` — Full spec
- ✅ `docs/FABRIC_HANDOFF_BLU.md` — Blu's quick-start

---

## Next Sync

- **Jose:** Get Foundry student endpoint, test eval script
- **Blu:** Set up Fabric tables, test path access
- **Both:** Quick Slack sync to confirm no blockers
- **Then:** Run full pipeline, verify scorecard

---

**Ready to go! 🚀**
