#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

RUN = "dev3-student-v1-text-eval-v2"
WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
BASE = f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_evals/{RUN}"
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "eval_run_name": RUN,
    "status": "current_for_blu",
    "do_not_mix_with_runs": [
        "dev3-baseline-20260623T185959",
        "dev3-teacher-anchored-clean",
        "dev3-student-v1-text-eval"
    ],
    "row_count_contract": {
        "eval_pool_size": 10,
        "eval_details_rows": 10,
        "baseline_rows": 10,
        "student_rows": 10,
        "teacher_rows": 10
    },
    "counts": {
        "baseline_correct": 1,
        "student_correct": 0,
        "teacher_correct": 10,
        "candidate_count_student_false_teacher_true": 10
    },
    "candidate_rule": "student=false AND teacher=true",
    "files": {
        "eval_results": "eval_results_2026-06-24T14-06-36.json",
        "eval_details": "eval_details_2026-06-24T14-06-41.jsonl"
    },
    "notes": "This v2 run is the only current run Blu should ingest for student-v1 retrain candidates. Older runs have different row counts by design."
}
fs = DataLakeServiceClient(account_url="https://onelake.dfs.fabric.microsoft.com", credential=AzureCliCredential()).get_file_system_client(WORKSPACE)
path = f"{BASE}/manifest.json"
fs.get_file_client(path).upload_data(json.dumps(manifest, indent=2).encode("utf-8"), overwrite=True)
print(f"Uploaded {path}")
