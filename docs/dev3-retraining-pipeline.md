# Dev3 Retraining Pipeline Runbook

This runbook documents the path that worked for the GPT-4.1-nano retraining demo.

## What This Pipeline Proves

The loop can:

1. Evaluate a deployed student against a teacher-gold held-out set.
2. Export `eval_results`, `eval_details`, `manifest`, and traces to Fabric.
3. Let Blu build a retraining set in OneLake.
4. Fine-tune the previous student model again using Blu's export.
5. Deploy the next candidate.
6. Evaluate and gate promotion.

A candidate can improve without being promotion-ready. That is expected and part of the gate.

## Current Working Format

Azure OpenAI preprocessing rejected native tool schema fine-tune files for GPT-4.1-nano GlobalStandard fine-tuning. The working format is schema-free text SFT:

```json
{"messages":[
  {"role":"system","content":"You are a TMG operations tool-calling agent..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"<tool_call>{\"name\":\"tool\",\"arguments\":{...}}</tool_call>"}
]}
```

The runtime parser already supports `<tool_call>{...}</tool_call>` output.

## Key Scripts

### One-loop retraining entrypoint

Use this when Blu has exported the next retraining folder:

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe scripts/run_dev3_retrain_once.py `
  --dataset-version dev3-student-v2 `
  --base-model "gpt-4.1-nano-2025-04-14.ft-56718845ff1c4fd1a68a50e7c7800f6d-dev3-text-account-v1" `
  --deployment-name gpt-41-nano-student-v2
```

What it does:

1. Downloads `Files/llmops/foundry_exports/<dataset-version>/train.jsonl`.
2. Converts Blu's rows to text SFT.
3. Uploads to the Azure OpenAI account endpoint.
4. Submits `trainingType=GlobalStandard` fine-tune.
5. Waits for completion.
6. Deploys the model.
7. Writes local audit files under `artifacts/<dataset-version>/`.

### Student-v2 eval export for Blu

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe scripts/eval_student_v2_full_for_blu.py
```

Writes:

```text
Files/llmops/raw/foundry_evals/dev3-student-v2-full-eval/
  eval_results.json
  eval_details.jsonl
  manifest.json

Files/llmops/raw/foundry_traces/traces_dev3_student_v2_<timestamp>.jsonl
```

### Dashboard reporting publisher

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe scripts/publish_executive_dashboard_tables.py
.venv\Scripts\python.exe scripts/upload_reporting_csv_only.py
```

Creates reporting-only files:

```text
Files/llmops/reporting/*.jsonl
Files/llmops/reporting_csv/*.csv
```

Then run this in a Fabric notebook attached to `lh_llmops` to create tables:

```python
TABLES = [
    "executive_scorecard_latest",
    "executive_kpis_latest",
    "executive_scorecard_history",
    "promotion_decisions",
]

for table in TABLES:
    path = f"Files/llmops/reporting_csv/{table}.csv"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    print(table, spark.table(table).count())
```

These are reporting-only tables and should not affect Blu's raw/silver/gold pipeline tables.

### Latency/RPS benchmark

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe scripts/benchmark_deployments_latency.py --sample-size 100 --concurrency 2 --seed 42
```

Latest clean result:

| Deployment | Success | Avg latency | P95 latency | RPS |
| --- | ---: | ---: | ---: | ---: |
| gpt-5.4 | 100/100 | 1.494s | 2.046s | 1.33 |
| gpt-41-nano-base | 100/100 | 1.377s | 1.842s | 1.45 |

Concurrency 5 is a stress test; nano hit rate limits there.

## Current Demo State

Student-v1:

```text
gpt-41-nano-student-v1
```

Student-v2:

```text
gpt-41-nano-student-v2
```

Student-v2 eval:

```text
student-v1 baseline: 0/10
student-v2:          1/10
teacher-gold:       10/10
promotion: rejected
```

This proves the loop moved the model but does not justify promotion.

## Fabric Folder Contract

Blu writes retraining data here:

```text
Files/llmops/foundry_exports/<dataset-version>/
  train.jsonl
  manifest.json
```

The eval exporter writes stable raw folders:

```text
Files/llmops/raw/foundry_evals/<run-name>/
  eval_results.json
  eval_details.jsonl
  manifest.json
```

Traces go here:

```text
Files/llmops/raw/foundry_traces/traces_<run-name>_<timestamp>.jsonl
```

Canonical report packages can use:

```text
Files/llmops/runs/<run-name>/
  eval_results.json
  eval_details.jsonl
  traces.jsonl
  eval_pool.jsonl
  manifest.json
```

## ACA Hosting Readiness

Deployment scaffold exists:

```text
Dockerfile
azure.yaml
infra/main.bicep
infra/main.parameters.json
.azure/deployment-plan.md
```

Recommended Azure hosting pattern:

```text
Azure Container Apps Job, manual trigger first, cron trigger later.
```

Before deployment, finish hardening `run_retrain_loop.py` or use `scripts/run_dev3_retrain_once.py` as the ACA job command.

## Promotion Gate

Promotion should require at minimum:

```text
student_accuracy >= 70%
student_accuracy >= previous_student_accuracy
```

Student-v2 should remain rejected because it scored only 1/10 on the hard teacher-gold eval.
