#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from collections import Counter

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

RUN = sys.argv[1] if len(sys.argv) > 1 else "dev3-student-v1-text-eval-v2"
WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
BASE = f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_evals/{RUN}"
fs = DataLakeServiceClient(
    account_url="https://onelake.dfs.fabric.microsoft.com",
    credential=AzureCliCredential(),
).get_file_system_client(WORKSPACE)
paths = list(fs.get_paths(path=BASE, recursive=False))
summary_path = next(p.name for p in paths if "eval_results_" in p.name)
detail_path = next(p.name for p in paths if "eval_details_" in p.name)
summary = json.loads(fs.get_file_client(summary_path).download_file().readall())
details = [json.loads(l) for l in fs.get_file_client(detail_path).download_file().readall().decode("utf-8").splitlines() if l.strip()]
counts = {}
for model in ["baseline", "student", "teacher"]:
    counts[model] = Counter(d["ast_match_by_model"].get(model) for d in details)
print(json.dumps({
    "run": RUN,
    "summary_eval_pool_size": summary.get("eval_pool_size"),
    "detail_rows": len(details),
    "model_true_false_counts": {m: {str(k): v for k, v in c.items()} for m, c in counts.items()},
    "candidate_count_student_false_teacher_true": sum(1 for d in details if (not d["ast_match_by_model"].get("student")) and d["ast_match_by_model"].get("teacher")),
    "summary_models": summary.get("models"),
}, indent=2))
