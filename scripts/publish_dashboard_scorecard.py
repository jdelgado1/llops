#!/usr/bin/env python
"""Publish dashboard-ready scorecard tables to OneLake.

Writes stable reporting files that Power BI can consume without joining raw eval
folders or summing percentage columns incorrectly:

  Files/llmops/reporting/scorecard_model_runs.jsonl
  Files/llmops/reporting/promotion_decisions.jsonl

Each scorecard row is one model role within one eval run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
REPORTING_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/reporting"

RUNS = [
    {
        "run_name": "dev3-student-v1-text-eval-v2",
        "path": f"{LAKEHOUSE}.Lakehouse/Files/llmops/runs/dev3-student-v1-text-eval-v2/eval_results.json",
        "iteration": 1,
        "candidate_model": "gpt-41-nano-student-v1",
        "previous_model": "gpt-41-nano-base",
        "status": "rejected",
        "decision_reason": "student-v1 scored 0/10 on hard teacher-gold eval",
    },
    {
        "run_name": "dev3-student-v2-full-eval",
        "path": f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_evals/dev3-student-v2-full-eval/eval_results.json",
        "iteration": 2,
        "candidate_model": "gpt-41-nano-student-v2",
        "previous_model": "gpt-41-nano-student-v1",
        "status": "rejected",
        "decision_reason": "student-v2 improved by 1 item but scored only 1/10; below promotion threshold",
    },
]

PROMOTION_THRESHOLD_PCT = 70.0


def download_json(fs, path: str) -> dict[str, Any]:
    return json.loads(fs.get_file_client(path).download_file().readall())


def upload_jsonl(fs, path: str, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    fs.get_directory_client("/".join(path.split("/")[:-1])).create_directory()
    fs.get_file_client(path).upload_data(content, overwrite=True)
    print(f"Uploaded {len(rows)} rows -> {path}")


def main() -> None:
    fs = DataLakeServiceClient(account_url=ACCOUNT_URL, credential=AzureCliCredential()).get_file_system_client(WORKSPACE)
    published_at = datetime.now(timezone.utc).isoformat()
    score_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for run in RUNS:
        result = download_json(fs, run["path"])
        eval_pool_size = int(result.get("eval_pool_size") or 0)
        candidate_count = int(result.get("candidate_count") or 0)
        models = result.get("models", {})
        student_acc = float(models.get("student", {}).get("ast_accuracy") or 0.0)
        baseline_acc = float(models.get("baseline", {}).get("ast_accuracy") or 0.0)
        teacher_acc = float(models.get("teacher", {}).get("ast_accuracy") or 100.0)
        improvement = student_acc - baseline_acc
        promoted = student_acc >= PROMOTION_THRESHOLD_PCT and student_acc >= baseline_acc

        for model_role in ["baseline", "student", "teacher"]:
            model = models.get(model_role, {})
            accuracy = float(model.get("ast_accuracy") or 0.0)
            num_correct = int(model.get("num_correct") if model.get("num_correct") is not None else round(accuracy / 100 * eval_pool_size))
            score_rows.append({
                "published_at": published_at,
                "run_name": run["run_name"],
                "iteration": run["iteration"],
                "model_role": model_role,
                "model_label": {
                    "baseline": run["previous_model"],
                    "student": run["candidate_model"],
                    "teacher": "teacher-gold",
                }[model_role],
                "ast_accuracy_pct": accuracy,
                "num_correct": num_correct,
                "num_incorrect": eval_pool_size - num_correct,
                "eval_pool_size": eval_pool_size,
                "candidate_count": candidate_count,
                "promotion_threshold_pct": PROMOTION_THRESHOLD_PCT,
                "student_vs_baseline_gap_pct": improvement if model_role == "student" else None,
                "teacher_gap_pct": (teacher_acc - accuracy),
                "promotion_status": "promoted" if promoted else "rejected",
                "decision_reason": run["decision_reason"],
            })

        decisions.append({
            "published_at": published_at,
            "run_name": run["run_name"],
            "iteration": run["iteration"],
            "candidate_model": run["candidate_model"],
            "previous_model": run["previous_model"],
            "teacher_accuracy_pct": teacher_acc,
            "previous_accuracy_pct": baseline_acc,
            "candidate_accuracy_pct": student_acc,
            "improvement_pct": improvement,
            "eval_pool_size": eval_pool_size,
            "candidate_count": candidate_count,
            "promotion_threshold_pct": PROMOTION_THRESHOLD_PCT,
            "promotion_status": "promoted" if promoted else "rejected",
            "decision_reason": run["decision_reason"],
        })

    upload_jsonl(fs, f"{REPORTING_DIR}/scorecard_model_runs.jsonl", score_rows)
    upload_jsonl(fs, f"{REPORTING_DIR}/promotion_decisions.jsonl", decisions)
    print(json.dumps({"scorecard_rows": len(score_rows), "decision_rows": len(decisions)}, indent=2))


if __name__ == "__main__":
    main()
