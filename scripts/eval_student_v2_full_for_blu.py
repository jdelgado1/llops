#!/usr/bin/env python
"""Run student-v2 held-out eval and export exactly in Blu's requested layout.

Writes:
  Files/llmops/raw/foundry_evals/dev3-student-v2-full-eval/
    eval_results.json
    eval_details.jsonl
    manifest.json

And traces:
  Files/llmops/raw/foundry_traces/traces_dev3_student_v2_<timestamp>.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.config import get_settings
from llmops.models import get_client
from llmops.tool_models import _parse_calls_from_text

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
RUN_NAME = "dev3-student-v2-full-eval"
EVAL_TEXT = ROOT / "artifacts" / "dev3-teacher-gold-small-text" / "eval.jsonl"
EVAL_SOURCE = ROOT / "artifacts" / "dev3-teacher-gold-small" / "eval.jsonl"
BASELINE_DEPLOYMENT = "gpt-41-nano-student-v1"
STUDENT_DEPLOYMENT = "gpt-41-nano-student-v2"
TEACHER_LABEL = "teacher"


def expected_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    assistant = next(m for m in record["messages"] if m.get("role") == "assistant")
    return _parse_calls_from_text(assistant.get("content"))


def normalize_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": c.get("name"), "arguments": c.get("arguments") or {}} for c in calls]


def calls_equal(pred: list[dict[str, Any]], ref: list[dict[str, Any]]) -> bool:
    return normalize_calls(pred) == normalize_calls(ref)


def invoke_text(client, deployment: str, prompt: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]], float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment,
        messages=prompt,
        max_completion_tokens=512,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw = response.choices[0].message.content or ""
    return raw, _parse_calls_from_text(raw), latency_ms


def upload(fs, path: str, content: bytes) -> None:
    fs.get_directory_client("/".join(path.split("/")[:-1])).create_directory()
    fs.get_file_client(path).upload_data(content, overwrite=True)
    print(f"Uploaded -> {path}")


def main() -> None:
    client = get_client(get_settings())
    rows = [json.loads(line) for line in EVAL_TEXT.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_rows = [json.loads(line) for line in EVAL_SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(source_rows):
        raise RuntimeError(f"row mismatch: text={len(rows)} source={len(source_rows)}")

    details: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    correct = {"baseline": 0, "student": 0, TEACHER_LABEL: len(rows)}

    for index, row in enumerate(rows):
        source = source_rows[index]
        source_meta = source.get("metadata", {})
        prompt = [m for m in row["messages"] if m.get("role") != "assistant"]
        user_message = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
        reference = expected_calls(row)

        baseline_raw, baseline_calls, baseline_latency = invoke_text(client, BASELINE_DEPLOYMENT, prompt)
        student_raw, student_calls, student_latency = invoke_text(client, STUDENT_DEPLOYMENT, prompt)
        baseline_ok = calls_equal(baseline_calls, reference)
        student_ok = calls_equal(student_calls, reference)
        correct["baseline"] += int(baseline_ok)
        correct["student"] += int(student_ok)

        details.append({
            "eval_item_id": index,
            "source_eval_item_id": source_meta.get("source_index", index),
            "eval_run_name": RUN_NAME,
            "dataset_version": "dev3-teacher-gold-small-text",
            "source_dataset_version": source_meta.get("dataset_version", "dev3-teacher-gold-small"),
            "source_pool": source_meta.get("source_pool"),
            "gold_source": source_meta.get("gold_source", "teacher_anchored_reference_call"),
            "teacher_model": source_meta.get("teacher_model", "gpt-5.4"),
            "baseline_deployment": BASELINE_DEPLOYMENT,
            "student_deployment": STUDENT_DEPLOYMENT,
            "user_message": user_message,
            "messages": prompt,
            "tools": source.get("tools", []),
            "reference_tool_calls": reference,
            "ast_match_by_model": {
                "baseline": baseline_ok,
                "student": student_ok,
                TEACHER_LABEL: True,
            },
            "predicted_calls_by_model": {
                "baseline": baseline_calls,
                "student": student_calls,
                TEACHER_LABEL: reference,
            },
            "raw_outputs_by_model": {
                "baseline": baseline_raw,
                "student": student_raw,
            },
            "latency_ms_by_model": {
                "baseline": baseline_latency,
                "student": student_latency,
            },
        })

        traces.append({
            "trace_id": f"{RUN_NAME}-{index}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_item_id": index,
            "source_eval_item_id": source_meta.get("source_index", index),
            "model": STUDENT_DEPLOYMENT,
            "messages": prompt,
            "raw_output": student_raw,
            "tool_calls": student_calls,
            "reference_tool_calls": reference,
            "ast_match": student_ok,
            "latency_ms": student_latency,
        })
        print(f"{index + 1}/{len(rows)} baseline={baseline_ok} student={student_ok}", flush=True)

    eval_pool_size = len(rows)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_run_name": RUN_NAME,
        "eval_pool_size": eval_pool_size,
        "teacher_gold_text_eval": True,
        "baseline_deployment": BASELINE_DEPLOYMENT,
        "student_deployment": STUDENT_DEPLOYMENT,
        "models": {
            "baseline": {
                "num_correct": correct["baseline"],
                "ast_accuracy": correct["baseline"] / eval_pool_size * 100,
            },
            "student": {
                "num_correct": correct["student"],
                "ast_accuracy": correct["student"] / eval_pool_size * 100,
            },
            TEACHER_LABEL: {
                "num_correct": correct[TEACHER_LABEL],
                "ast_accuracy": 100.0,
            },
        },
        "candidate_rule": "student=false AND teacher=true",
        "candidate_count": sum(1 for d in details if not d["ast_match_by_model"]["student"] and d["ast_match_by_model"][TEACHER_LABEL]),
    }
    manifest = {
        "run_name": RUN_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "raw-eval-package-v1",
        "files": {
            "eval_results": "eval_results.json",
            "eval_details": "eval_details.jsonl",
            "manifest": "manifest.json",
            "traces": "Files/llmops/raw/foundry_traces/traces_dev3_student_v2_<timestamp>.jsonl",
        },
        "row_count_contract": {
            "eval_pool_rows": eval_pool_size,
            "eval_details_rows": len(details),
            "baseline_rows": len(details),
            "student_rows": len(details),
            "teacher_rows": len(details),
        },
        "candidate_rule": summary["candidate_rule"],
        "candidate_count": summary["candidate_count"],
        "notes": "Blu-requested stable raw eval folder for student-v2.",
    }

    fs = DataLakeServiceClient(account_url=ACCOUNT_URL, credential=AzureCliCredential()).get_file_system_client(WORKSPACE)
    eval_dir = f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_evals/{RUN_NAME}"
    trace_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = f"{LAKEHOUSE}.Lakehouse/Files/llmops/raw/foundry_traces/traces_dev3_student_v2_{trace_stamp}.jsonl"

    upload(fs, f"{eval_dir}/eval_results.json", json.dumps(summary, indent=2).encode("utf-8"))
    upload(fs, f"{eval_dir}/eval_details.jsonl", "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details).encode("utf-8"))
    upload(fs, f"{eval_dir}/manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    upload(fs, trace_path, "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces).encode("utf-8"))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Eval folder: Files/llmops/raw/foundry_evals/{RUN_NAME}/", flush=True)
    print(f"Trace file: {trace_path.replace(LAKEHOUSE + '.Lakehouse/', '')}", flush=True)


if __name__ == "__main__":
    main()
