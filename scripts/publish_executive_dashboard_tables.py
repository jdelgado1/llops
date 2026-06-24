#!/usr/bin/env python
"""Publish simplified Power BI executive dashboard tables under Files/llmops/reporting only.

This intentionally does NOT touch raw evals, traces, or Blu's foundry_exports.

Files:
  Files/llmops/reporting/executive_scorecard_latest.jsonl
  Files/llmops/reporting/executive_scorecard_history.jsonl
  Files/llmops/reporting/executive_kpis_latest.jsonl
  Files/llmops/reporting/dashboard_manifest.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
REPORTING_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/reporting"
ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"

# Dashboard contract: keep latest scorecard to one row per role.
LATEST_RUN = {
    "run_name": "dev3-student-v2-full-eval",
    "iteration": 2,
    "baseline_label": "student-v1",
    "student_label": "student-v2",
    "teacher_label": "teacher-gold",
    "baseline_accuracy": 0.0,
    "student_accuracy": 10.0,
    "teacher_accuracy": 100.0,
    "student_correct": 1,
    "baseline_correct": 0,
    "teacher_correct": 10,
    "eval_pool_size": 10,
    "candidate_count": 9,
    "promotion_threshold": 70.0,
    "promotion_status": "rejected",
    "decision_reason": "Loop improved from 0/10 to 1/10, but 10% is below the 70% promotion threshold.",
}

HISTORY = [
    {
        "run_name": "dev3-student-v1-text-eval-v2",
        "iteration": 1,
        "baseline_label": "base-nano",
        "student_label": "student-v1",
        "teacher_label": "teacher-gold",
        "baseline_accuracy": 10.0,
        "student_accuracy": 0.0,
        "teacher_accuracy": 100.0,
        "student_correct": 0,
        "baseline_correct": 1,
        "teacher_correct": 10,
        "eval_pool_size": 10,
        "candidate_count": 10,
        "promotion_threshold": 70.0,
        "promotion_status": "rejected",
        "decision_reason": "Seed model scored 0/10 on hard teacher-gold eval.",
    },
    LATEST_RUN,
]


def score_rows(run: dict) -> list[dict]:
    published_at = datetime.now(timezone.utc).isoformat()
    gap = run["student_accuracy"] - run["baseline_accuracy"]
    common = {
        "published_at": published_at,
        "run_name": run["run_name"],
        "iteration": run["iteration"],
        "eval_pool_size": run["eval_pool_size"],
        "candidate_count": run["candidate_count"],
        "promotion_status": run["promotion_status"],
        "decision_reason": run["decision_reason"],
    }
    return [
        {
            **common,
            "model_role": "baseline",
            "model_label": run["baseline_label"],
            "ast_accuracy": run["baseline_accuracy"],
            "ast_accuracy_pct": run["baseline_accuracy"],
            "num_correct": run["baseline_correct"],
            "minimum_acceptable_ast": None,
            "student_ast_accuracy": None,
            "student_vs_baseline_gap": None,
        },
        {
            **common,
            "model_role": "student",
            "model_label": run["student_label"],
            "ast_accuracy": run["student_accuracy"],
            "ast_accuracy_pct": run["student_accuracy"],
            "num_correct": run["student_correct"],
            # Cards in current report are using SUM, so only populate these once.
            "minimum_acceptable_ast": run["promotion_threshold"],
            "student_ast_accuracy": run["student_accuracy"],
            "student_vs_baseline_gap": gap,
        },
        {
            **common,
            "model_role": "teacher",
            "model_label": run["teacher_label"],
            "ast_accuracy": run["teacher_accuracy"],
            "ast_accuracy_pct": run["teacher_accuracy"],
            "num_correct": run["teacher_correct"],
            "minimum_acceptable_ast": None,
            "student_ast_accuracy": None,
            "student_vs_baseline_gap": None,
        },
    ]


def kpi_row(run: dict) -> dict:
    gap = run["student_accuracy"] - run["baseline_accuracy"]
    return {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run["run_name"],
        "iteration": run["iteration"],
        "minimum_acceptable_ast": run["promotion_threshold"],
        "student_ast_accuracy": run["student_accuracy"],
        "baseline_ast_accuracy": run["baseline_accuracy"],
        "teacher_ast_accuracy": run["teacher_accuracy"],
        "student_vs_baseline_gap": gap,
        "eval_pool_size": run["eval_pool_size"],
        "student_correct": run["student_correct"],
        "candidate_count": run["candidate_count"],
        "promotion_status": run["promotion_status"],
        "decision_reason": run["decision_reason"],
    }


def upload_jsonl(fs, name: str, rows: list[dict]) -> None:
    path = f"{REPORTING_DIR}/{name}"
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    fs.get_directory_client(REPORTING_DIR).create_directory()
    fs.get_file_client(path).upload_data(content, overwrite=True)
    print(f"Uploaded {len(rows)} rows -> {path}")


def upload_json(fs, name: str, obj: dict) -> None:
    path = f"{REPORTING_DIR}/{name}"
    fs.get_directory_client(REPORTING_DIR).create_directory()
    fs.get_file_client(path).upload_data(json.dumps(obj, indent=2).encode("utf-8"), overwrite=True)
    print(f"Uploaded -> {path}")


def main() -> None:
    fs = DataLakeServiceClient(account_url=ACCOUNT_URL, credential=AzureCliCredential()).get_file_system_client(WORKSPACE)
    latest_rows = score_rows(LATEST_RUN)
    history_rows = [row for run in HISTORY for row in score_rows(run)]
    upload_jsonl(fs, "executive_scorecard_latest.jsonl", latest_rows)
    upload_jsonl(fs, "executive_scorecard_history.jsonl", history_rows)
    upload_jsonl(fs, "executive_kpis_latest.jsonl", [kpi_row(LATEST_RUN)])
    upload_json(fs, "dashboard_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recommended_power_bi_tables": {
            "latest_bar_and_cards": "executive_scorecard_latest.jsonl",
            "single_row_cards": "executive_kpis_latest.jsonl",
            "trend_line": "executive_scorecard_history.jsonl",
        },
        "notes": [
            "Use executive_scorecard_latest for the bar chart: model_role vs ast_accuracy.",
            "Use executive_kpis_latest for cards to avoid Power BI summing percentage values across role rows.",
            "Do not use fields with model_role values like teacher_on_filtered_slice for the executive page."
        ],
    })


if __name__ == "__main__":
    main()
