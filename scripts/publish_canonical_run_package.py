#!/usr/bin/env python
"""Publish one canonical run folder with stable filenames for Blu.

Folder shape:
  Files/llmops/runs/<run_name>/
    eval_results.json
    eval_details.jsonl
    traces.jsonl
    eval_pool.jsonl
    manifest.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
RUN_NAME = "dev3-student-v1-text-eval-v2"
SRC_EVAL_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_evals/{RUN_NAME}"
DEST_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/runs/{RUN_NAME}"
LOCAL_EVAL_POOL = Path("artifacts/dev3-teacher-gold-small-text/eval.jsonl")
LOCAL_TRACES = Path("artifacts/real_traces_student.jsonl")


def latest(paths: list[str], prefix: str) -> str:
    candidates = sorted(p for p in paths if Path(p).name.startswith(prefix))
    if not candidates:
        raise FileNotFoundError(prefix)
    return candidates[-1]


def upload(fs, path: str, content: bytes) -> None:
    fs.get_directory_client("/".join(path.split("/")[:-1])).create_directory()
    fs.get_file_client(path).upload_data(content, overwrite=True)
    print(f"Uploaded -> {path}")


def main() -> None:
    fs = DataLakeServiceClient(account_url=ACCOUNT_URL, credential=AzureCliCredential()).get_file_system_client(WORKSPACE)
    src_paths = [p.name for p in fs.get_paths(path=SRC_EVAL_DIR, recursive=False)]
    src_results = latest(src_paths, "eval_results_")
    src_details = latest(src_paths, "eval_details_")
    src_manifest = next((p for p in src_paths if Path(p).name == "manifest.json"), None)

    eval_results = fs.get_file_client(src_results).download_file().readall()
    eval_details = fs.get_file_client(src_details).download_file().readall()
    eval_pool = LOCAL_EVAL_POOL.read_bytes()
    traces = LOCAL_TRACES.read_bytes() if LOCAL_TRACES.exists() else b""

    source_manifest = {}
    if src_manifest:
        source_manifest = json.loads(fs.get_file_client(src_manifest).download_file().readall())

    manifest = {
        "run_name": RUN_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "run-package-v1",
        "purpose": "student_v1_teacher_gold_text_eval",
        "folder": f"Files/llmops/runs/{RUN_NAME}/",
        "files": {
            "eval_results": "eval_results.json",
            "eval_details": "eval_details.jsonl",
            "traces": "traces.jsonl",
            "eval_pool": "eval_pool.jsonl",
            "manifest": "manifest.json",
        },
        "row_count_contract": {
            "eval_pool_rows": 10,
            "eval_details_rows": 10,
            "baseline_rows": 10,
            "student_rows": 10,
            "teacher_rows": 10,
        },
        "candidate_rule": "student=false AND teacher=true",
        "source_eval_folder": f"Files/llmops/raw/foundry_evals/{RUN_NAME}/",
        "source_manifest": source_manifest,
        "notes": "Canonical package for Blu. Use this folder only; do not join with raw timestamped folders from earlier experiments.",
    }

    upload(fs, f"{DEST_DIR}/eval_results.json", eval_results)
    upload(fs, f"{DEST_DIR}/eval_details.jsonl", eval_details)
    upload(fs, f"{DEST_DIR}/traces.jsonl", traces)
    upload(fs, f"{DEST_DIR}/eval_pool.jsonl", eval_pool)
    upload(fs, f"{DEST_DIR}/manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    print(f"Canonical package ready: Files/llmops/runs/{RUN_NAME}/")


if __name__ == "__main__":
    main()
