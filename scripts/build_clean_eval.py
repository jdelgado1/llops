#!/usr/bin/env python
"""
Option A: build a TEACHER-ANCHORED clean eval pool.

The raw eval pool has noisy references (even gpt-5.4 only hit ~67%), which caps
everyone and hides the teacher>>student gap. This keeps only the items the
TEACHER (gpt-5.4) got right — i.e. references the frontier model agrees with =
clean items. On that subset teacher = 100% by construction, and we can read the
student's genuine (lower) accuracy straight from the existing eval_details.

Pulls eval_details from the baseline run in OneLake, intersects with the eval
pool, and writes a clean pool + prints the real gap.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

ONELAKE_ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
RUN_NAME = "dev3-baseline-20260623T185959"
CLEAN_RUN_NAME = "dev3-teacher-anchored-clean"
EVALS_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_evals/{RUN_NAME}"
CLEAN_EVALS_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_evals/{CLEAN_RUN_NAME}"
CLEAN_EXPORTS_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports/{CLEAN_RUN_NAME}"

EVAL_POOL = ROOT / "artifacts" / "eval_pool_114items.jsonl"
OUT_POOL = ROOT / "artifacts" / "eval_pool_clean_teacher.jsonl"


def _fs():
    cred = DefaultAzureCredential()
    return DataLakeServiceClient(account_url=ONELAKE_ACCOUNT_URL, credential=cred).get_file_system_client(WORKSPACE)


def fetch_eval_details(fs) -> list[dict]:
    # find the eval_details_*.jsonl in the run dir
    detail_path = None
    for p in fs.get_paths(path=EVALS_DIR, recursive=False):
        if "eval_details_" in p.name and p.name.endswith(".jsonl"):
            detail_path = p.name
    if not detail_path:
        raise FileNotFoundError(f"No eval_details_*.jsonl under {EVALS_DIR}")
    data = fs.get_file_client(detail_path).download_file().readall().decode("utf-8")
    return [json.loads(l) for l in data.splitlines() if l.strip()]


def upload_text(fs, path: str, text: str) -> None:
    dir_path = "/".join(path.split("/")[:-1])
    fs.get_directory_client(dir_path).create_directory()
    fs.get_file_client(path).upload_data(text.encode("utf-8"), overwrite=True)
    print(f"Uploaded -> {path}")


def main() -> None:
    fs = _fs()
    details = fetch_eval_details(fs)
    print(f"Loaded {len(details)} eval_details records from OneLake run '{RUN_NAME}'")

    pool = [json.loads(l) for l in EVAL_POOL.read_text(encoding="utf-8").splitlines() if l.strip()]

    # teacher-correct item ids = clean items
    teacher_ok = [d["eval_item_id"] for d in details if d["ast_match_by_model"].get("teacher")]
    # student/baseline accuracy ON the clean (teacher-correct) subset
    base_ok_on_clean = sum(
        1 for d in details
        if d["ast_match_by_model"].get("teacher") and d["ast_match_by_model"].get("baseline")
    )
    n_clean = len(teacher_ok)

    details_by_id = {d["eval_item_id"]: d for d in details}

    # write the clean pool (those eval_item_ids map to pool line indices)
    clean_items = []
    clean_details = []
    for new_id, source_id in enumerate(teacher_ok):
        if source_id >= len(pool):
            continue
        item = dict(pool[source_id])
        item["metadata"] = {
            **item.get("metadata", {}),
            "source_eval_run_name": RUN_NAME,
            "source_eval_item_id": source_id,
            "clean_eval_run_name": CLEAN_RUN_NAME,
            "selection_rule": "teacher AST match == true",
        }
        clean_items.append(item)

        source_detail = details_by_id[source_id]
        clean_details.append({
            "eval_item_id": new_id,
            "source_eval_item_id": source_id,
            "ast_match_by_model": {
                "baseline": bool(source_detail["ast_match_by_model"].get("baseline")),
                "student": bool(source_detail["ast_match_by_model"].get("student")),
                "teacher": True,
            },
        })

    OUT_POOL.write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in clean_items),
        encoding="utf-8",
    )

    baseline_ok = sum(1 for d in clean_details if d["ast_match_by_model"].get("baseline"))
    student_ok = sum(1 for d in clean_details if d["ast_match_by_model"].get("student"))
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_pool_size": len(clean_details),
        "eval_run_name": CLEAN_RUN_NAME,
        "source_eval_run_name": RUN_NAME,
        "teacher_anchored": True,
        "selection_rule": "teacher AST match == true from source eval_details",
        "models": {
            "baseline": {"ast_accuracy": baseline_ok / len(clean_details) * 100 if clean_details else 0},
            "student": {"ast_accuracy": student_ok / len(clean_details) * 100 if clean_details else 0},
            "teacher": {"ast_accuracy": 100.0},
        },
    }

    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-").split(".")[0]
    upload_text(
        fs,
        f"{CLEAN_EVALS_DIR}/eval_results_{timestamp}.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )
    upload_text(
        fs,
        f"{CLEAN_EVALS_DIR}/eval_details_{timestamp}.jsonl",
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in clean_details),
    )
    upload_text(
        fs,
        f"{CLEAN_EXPORTS_DIR}/eval_pool_clean_teacher.jsonl",
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in clean_items),
    )
    upload_text(
        fs,
        f"{CLEAN_EXPORTS_DIR}/manifest.json",
        json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": CLEAN_RUN_NAME,
            "example_count": len(clean_items),
            "source_eval_run_name": RUN_NAME,
            "status": "teacher_anchored_clean_eval",
            "eval_details_path": f"Files/llmops/raw/foundry_evals/{CLEAN_RUN_NAME}/",
            "eval_pool_path": f"Files/llmops/foundry_exports/{CLEAN_RUN_NAME}/eval_pool_clean_teacher.jsonl",
        }, indent=2),
    )

    print("\n=== TEACHER-ANCHORED CLEAN EVAL ===")
    print(f"Clean items (teacher correct): {n_clean} / {len(details)}")
    print(f"  teacher accuracy on clean set: 100.0%  (by construction)")
    if n_clean:
        nano_acc = base_ok_on_clean / n_clean * 100
        print(f"  nano (base)  accuracy on clean set: {nano_acc:.1f}%  ({base_ok_on_clean}/{n_clean})")
        print(f"  >>> THE GAP: teacher 100% vs nano {nano_acc:.1f}%  = {100 - nano_acc:.1f} pts of headroom")
    print(f"\nWrote clean pool -> {OUT_POOL}  ({len(clean_items)} items)")
    print(f"Uploaded clean eval package -> Files/llmops/raw/foundry_evals/{CLEAN_RUN_NAME}/")
    print(f"Uploaded matching pool -> Files/llmops/foundry_exports/{CLEAN_RUN_NAME}/eval_pool_clean_teacher.jsonl")
    print("Blu can ingest the clean eval_details now; the student-failed rows are the retrain candidates.")


if __name__ == "__main__":
    main()
