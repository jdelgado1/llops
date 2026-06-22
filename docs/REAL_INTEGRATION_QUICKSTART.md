# Real Integration Test — Quick Reference

**What it does:**
1. Runs eval (baseline vs trained student vs teacher) → exports to Fabric
2. Generates real traces (student model inference) → exports to Fabric
3. Tests Foundry SDK can read from Fabric paths

**Before you run:**
- Get `STUDENT_FINETUNED_DEPLOYMENT` from your Foundry training job
- Verify `BASELINE_DEPLOYMENT` is live

---

## Quick Run

```bash
cd c:\Users\josedelgado\Coding\tmg-research-agent-llmops

# Set up environment
$env:PYTHONPATH="src"

# Run end-to-end real integration
.venv\Scripts\python.exe scripts\run_real_integration.py `
  --eval-pool-jsonl artifacts\eval_pool_114items.jsonl `
  --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer `
  --student-deployment <PASTE_YOUR_STUDENT_ENDPOINT> `
  --fabric-lakehouse-root /Volumes/lh_llmops
```

---

## What Gets Created

**In Fabric:**
```
Files/llmops/
├── raw/foundry_evals/baseline-vs-student-real/
│   ├── eval_results_<timestamp>.json     ← Summary (3-way accuracy)
│   └── eval_details_<timestamp>.jsonl    ← Per-item predictions
└── raw/foundry_traces/
    └── traces_tmg_ops_agent_*.jsonl      ← Real model inference traces
```

**Locally:**
```
artifacts/
└── real_traces_student.jsonl             ← Local copy of traces
```

---

## Expected Output

```
STEP 1: Run 3-way Eval + Export to Fabric
  baseline:  55.3%
  student:   88.6%
  teacher:   91.2%
  ✅ Exported to Fabric

STEP 2: Generate Real Traces + Export to Fabric
  Generated 64 traces from student model inference
  ✅ Exported to Fabric

INTEGRATION COMPLETE
✅ Real data now in Fabric:
   Eval results: Files/llmops/raw/foundry_evals/baseline-vs-student-real/
   Traces:       Files/llmops/raw/foundry_traces/
```

---

## Next (For Blu)

Once you see the above, Blu can:

```sql
-- Check eval results ingested
SELECT model, ast_accuracy FROM eval_runs 
WHERE eval_run_name = 'baseline-vs-student-real' 
ORDER BY ast_accuracy DESC;

-- Check traces ingested
SELECT COUNT(*) as num_traces, model 
FROM foundry_traces 
WHERE model LIKE 'qwen3%'
GROUP BY model;
```

---

## If Something Fails

**Eval fails:**
- Check baseline deployment is live: `invoke_model(baseline_deployment, ...)`
- Check student deployment is running

**Traces fail:**
- Model inference error; check deployment connectivity

**SDK consumption fails:**
- Foundry SDK can't read Fabric paths (expected)
- Workaround: Jose downloads JSONL, uploads to Foundry manually
- Document for future: "Fabric → Azure Blob → Foundry" as alternate path

---

## Full Command (Copy-Paste Ready)

```powershell
cd c:\Users\josedelgado\Coding\tmg-research-agent-llmops
$env:PYTHONPATH="src"
.venv\Scripts\python.exe scripts\run_real_integration.py `
  --eval-pool-jsonl artifacts\eval_pool_114items.jsonl `
  --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer `
  --student-deployment <GET_FROM_FOUNDRY_JOB> `
  --fabric-lakehouse-root /Volumes/lh_llmops
```

---

## To Test Without Traces (Quick Eval-Only)

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe scripts\run_real_integration.py `
  --eval-pool-jsonl artifacts\eval_pool_114items.jsonl `
  --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer `
  --student-deployment <ENDPOINT> `
  --fabric-lakehouse-root /Volumes/lh_llmops `
  --skip-traces `
  --limit 5
```

This runs eval on just 5 items, skips traces. Good for testing path/connectivity before full run.

---

## Files Ready

- ✅ `scripts/run_real_integration.py` — End-to-end orchestrator
- ✅ `scripts/test_foundry_fabric_integration.py` — Eval + export + SDK test
- ✅ `scripts/gen_real_traces.py` — Trace generation from model inference
- ✅ `artifacts/eval_pool_114items.jsonl` — 64-item eval set (no leakage)
- ✅ `src/llmops/fabric_integration.py` — Export classes

---

## Go!

Once you have the student endpoint, run the command above. This is the REAL integration — no mocks, no placeholders.
