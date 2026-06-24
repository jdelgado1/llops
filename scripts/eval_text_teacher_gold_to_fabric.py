#!/usr/bin/env python
"""Evaluate text-tool-call student-v1 on held-out teacher-gold eval and upload to Fabric.

The fine-tuned model learned to emit <tool_call>{...}</tool_call> text, so we do
not pass tool schemas to the chat API. We parse text with the existing fallback
parser and compare against the held-out teacher-gold assistant content.
"""
from __future__ import annotations

import json
import re
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.config import get_settings
from llmops.fabric_integration import FabricExporter
from llmops.models import get_client
from llmops.tool_models import _parse_calls_from_text

EVAL = ROOT / "artifacts" / "dev3-teacher-gold-small-text" / "eval.jsonl"
SOURCE_EVAL = ROOT / "artifacts" / "dev3-teacher-gold-small" / "eval.jsonl"
RUN_NAME = "dev3-student-v1-text-eval-v2"
WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
BASE_DEPLOYMENT = "gpt-41-nano-base"
STUDENT_DEPLOYMENT = "gpt-41-nano-student-v1"
TEACHER_LABEL = "teacher"


def upload_canonical_run(exporter: FabricExporter, summary: dict[str, Any], details: list[dict[str, Any]]) -> None:
    """Write stable filenames under Files/llmops/runs/<run_name>/ for Fabric users."""
    run_dir = Path(f"onelake://{WORKSPACE}/{LAKEHOUSE}.Lakehouse/Files/llmops/runs/{RUN_NAME}")
    eval_pool_bytes = EVAL.read_bytes()
    traces_path = ROOT / "artifacts" / "real_traces_student.jsonl"
    traces_bytes = traces_path.read_bytes() if traces_path.exists() else b""
    manifest = {
        "run_name": RUN_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "run-package-v1",
        "files": {
            "eval_results": "eval_results.json",
            "eval_details": "eval_details.jsonl",
            "traces": "traces.jsonl",
            "eval_pool": "eval_pool.jsonl",
            "manifest": "manifest.json",
        },
        "row_count_contract": {
            "eval_pool_rows": summary["eval_pool_size"],
            "eval_details_rows": len(details),
            "baseline_rows": len(details),
            "student_rows": len(details),
            "teacher_rows": len(details),
        },
        "candidate_rule": summary.get("candidate_rule"),
        "candidate_count": summary.get("candidate_count"),
        "notes": "Canonical stable-file package. Prefer this over timestamped raw folders.",
    }
    exporter._write_file(run_dir / "eval_results.json", json.dumps(summary, indent=2).encode("utf-8"))
    exporter._write_file(run_dir / "eval_details.jsonl", "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details).encode("utf-8"))
    exporter._write_file(run_dir / "traces.jsonl", traces_bytes)
    exporter._write_file(run_dir / "eval_pool.jsonl", eval_pool_bytes)
    exporter._write_file(run_dir / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))


def expected_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    assistant = next(m for m in record["messages"] if m.get("role") == "assistant")
    return _parse_calls_from_text(assistant.get("content"))


def normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    return {"name": call.get("name"), "arguments": call.get("arguments") or {}}


def calls_equal(pred: list[dict[str, Any]], ref: list[dict[str, Any]]) -> bool:
    return [normalize_call(c) for c in pred] == [normalize_call(c) for c in ref]


def invoke_text(client, deployment: str, prompt_messages: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]], float]:
    started = time.perf_counter()
    resp = client.chat.completions.create(model=deployment, messages=prompt_messages, max_completion_tokens=512)
    latency_ms = (time.perf_counter() - started) * 1000
    content = resp.choices[0].message.content or ""
    return content, _parse_calls_from_text(content), latency_ms


def main() -> None:
    global RUN_NAME, BASE_DEPLOYMENT, STUDENT_DEPLOYMENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--baseline-deployment", default=BASE_DEPLOYMENT)
    parser.add_argument("--student-deployment", default=STUDENT_DEPLOYMENT)
    args = parser.parse_args()
    RUN_NAME = args.run_name
    BASE_DEPLOYMENT = args.baseline_deployment
    STUDENT_DEPLOYMENT = args.student_deployment

    client = get_client(get_settings())
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_rows = [json.loads(line) for line in SOURCE_EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(source_rows):
        raise RuntimeError(f"Text eval rows ({len(rows)}) != source eval rows ({len(source_rows)})")
    details = []
    correct = {"baseline": 0, "student": 0, TEACHER_LABEL: len(rows)}
    latency = {"baseline": [], "student": []}

    for i, row in enumerate(rows):
        source_row = source_rows[i]
        source_meta = source_row.get("metadata", {})
        prompt = [m for m in row["messages"] if m.get("role") != "assistant"]
        user_message = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
        ref = expected_calls(row)
        base_raw, base_calls, base_ms = invoke_text(client, BASE_DEPLOYMENT, prompt)
        student_raw, student_calls, student_ms = invoke_text(client, STUDENT_DEPLOYMENT, prompt)
        base_ok = calls_equal(base_calls, ref)
        student_ok = calls_equal(student_calls, ref)
        correct["baseline"] += int(base_ok)
        correct["student"] += int(student_ok)
        latency["baseline"].append(base_ms)
        latency["student"].append(student_ms)
        details.append({
            "eval_item_id": i,
            "source_eval_item_id": source_meta.get("source_index", i),
            "dataset_version": "dev3-teacher-gold-small-text",
            "source_dataset_version": source_meta.get("dataset_version", "dev3-teacher-gold-small"),
            "source_pool": source_meta.get("source_pool"),
            "gold_source": source_meta.get("gold_source", "teacher_anchored_reference_call"),
            "teacher_model": source_meta.get("teacher_model", "gpt-5.4"),
            "user_message": user_message,
            "messages": prompt,
            "tools": source_row.get("tools", []),
            "ast_match_by_model": {"baseline": base_ok, "student": student_ok, TEACHER_LABEL: True},
            "reference_tool_calls": ref,
            "predicted_calls_by_model": {"baseline": base_calls, "student": student_calls, TEACHER_LABEL: ref},
            "raw_outputs_by_model": {"baseline": base_raw, "student": student_raw},
            "latency_ms_by_model": {"baseline": base_ms, "student": student_ms},
        })
        print(f"{i+1}/{len(rows)} baseline={base_ok} student={student_ok}", flush=True)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_pool_size": len(rows),
        "eval_run_name": RUN_NAME,
        "teacher_gold_text_eval": True,
        "source_eval_file": str(SOURCE_EVAL.relative_to(ROOT)),
        "text_eval_file": str(EVAL.relative_to(ROOT)),
        "models": {
            "baseline": {"ast_accuracy": correct["baseline"] / len(rows) * 100, "num_correct": correct["baseline"]},
            "student": {"ast_accuracy": correct["student"] / len(rows) * 100, "num_correct": correct["student"]},
            TEACHER_LABEL: {"ast_accuracy": 100.0},
        },
        "candidate_rule": "student=false AND teacher=true",
        "candidate_count": len(rows) - correct["student"],
    }

    exporter = FabricExporter(onelake_workspace=WORKSPACE, onelake_lakehouse=LAKEHOUSE)
    exporter.export_eval_results(summary, RUN_NAME)
    exporter.export_eval_details_jsonl(details, RUN_NAME)
    upload_canonical_run(exporter, summary, details)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Uploaded eval package -> Files/llmops/raw/foundry_evals/{RUN_NAME}/", flush=True)
    print(f"Uploaded canonical package -> Files/llmops/runs/{RUN_NAME}/", flush=True)


if __name__ == "__main__":
    main()
