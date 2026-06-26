#!/usr/bin/env python
"""Estimate cost for the student-v2 held-out eval and training jobs.

Re-runs the 10-row held-out eval with usage capture because the prior eval export
kept latency/raw outputs but not token usage.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.config import get_settings
from llmops.models import get_client
from llmops.tool_models import _parse_calls_from_text

EVAL_TEXT = ROOT / "artifacts" / "dev3-teacher-gold-small-text" / "eval.jsonl"
BASELINE_DEPLOYMENT = "gpt-41-nano-student-v1"
STUDENT_DEPLOYMENT = "gpt-41-nano-student-v2"

# Azure OpenAI GPT-4.1-nano list-price estimates. Adjust if your offer differs.
# Common public pricing: input $0.10 / 1M tokens, output $0.40 / 1M tokens.
INPUT_PER_M = 0.10
OUTPUT_PER_M = 0.40
# Fine-tuning training token cost varies by region/offer. Keep explicit as an estimate.
# Public OpenAI-style reference is often around $1.50 / 1M training tokens for nano-class fine-tuning.
TRAINING_PER_M_EST = 1.50

TRAINED_TOKENS = {
    "student_v1": 6068,
    "student_v2": 2998,
}


def expected_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    assistant = next(m for m in record["messages"] if m.get("role") == "assistant")
    return _parse_calls_from_text(assistant.get("content"))


def normalize_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": c.get("name"), "arguments": c.get("arguments") or {}} for c in calls]


def calls_equal(pred: list[dict[str, Any]], ref: list[dict[str, Any]]) -> bool:
    return normalize_calls(pred) == normalize_calls(ref)


def invoke(client, deployment: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    started = time.perf_counter()
    resp = client.chat.completions.create(model=deployment, messages=messages, max_completion_tokens=512)
    latency_ms = (time.perf_counter() - started) * 1000
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return {
        "raw": content,
        "calls": _parse_calls_from_text(content),
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": (prompt_tokens / 1_000_000 * INPUT_PER_M) + (completion_tokens / 1_000_000 * OUTPUT_PER_M),
    }


def main() -> None:
    rows = [json.loads(line) for line in EVAL_TEXT.read_text(encoding="utf-8").splitlines() if line.strip()]
    client = get_client(get_settings())
    totals = {
        "baseline": {"correct": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "latencies": []},
        "student": {"correct": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "latencies": []},
    }
    details = []
    for idx, row in enumerate(rows):
        prompt = [m for m in row["messages"] if m.get("role") != "assistant"]
        ref = expected_calls(row)
        base = invoke(client, BASELINE_DEPLOYMENT, prompt)
        student = invoke(client, STUDENT_DEPLOYMENT, prompt)
        base_ok = calls_equal(base["calls"], ref)
        student_ok = calls_equal(student["calls"], ref)
        for label, result, ok in [("baseline", base, base_ok), ("student", student, student_ok)]:
            totals[label]["correct"] += int(ok)
            totals[label]["prompt_tokens"] += result["prompt_tokens"]
            totals[label]["completion_tokens"] += result["completion_tokens"]
            totals[label]["total_tokens"] += result["total_tokens"]
            totals[label]["cost"] += result["estimated_cost_usd"]
            totals[label]["latencies"].append(result["latency_ms"])
        details.append({
            "eval_item_id": idx,
            "baseline_correct": base_ok,
            "student_correct": student_ok,
            "baseline_usage": {k: base[k] for k in ["prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd", "latency_ms"]},
            "student_usage": {k: student[k] for k in ["prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd", "latency_ms"]},
        })
        print(f"{idx+1}/{len(rows)} baseline={base_ok} student={student_ok}", flush=True)

    summary = {
        "pricing_assumption_usd_per_1m_tokens": {
            "gpt_4_1_nano_input": INPUT_PER_M,
            "gpt_4_1_nano_output": OUTPUT_PER_M,
            "gpt_4_1_nano_training_estimate": TRAINING_PER_M_EST,
        },
        "eval_rows": len(rows),
        "models": {},
        "fine_tuning_training_estimate_usd": {},
        "promotion_verdict": "do_not_promote_student_v2",
        "promotion_reason": "student-v2 improved from 0/10 to 1/10 on hard held-out teacher-gold eval, but 10% accuracy is not production-ready.",
    }
    for label, total in totals.items():
        latencies = total["latencies"]
        summary["models"][label] = {
            "correct": total["correct"],
            "accuracy_pct": total["correct"] / len(rows) * 100,
            "prompt_tokens": total["prompt_tokens"],
            "completion_tokens": total["completion_tokens"],
            "total_tokens": total["total_tokens"],
            "estimated_inference_cost_usd": total["cost"],
            "avg_latency_ms": sum(latencies) / len(latencies),
        }
    for name, tokens in TRAINED_TOKENS.items():
        summary["fine_tuning_training_estimate_usd"][name] = {
            "trained_tokens": tokens,
            "estimated_training_cost_usd": tokens / 1_000_000 * TRAINING_PER_M_EST,
        }

    out = ROOT / "artifacts" / "student_v2_cost_summary.json"
    out.write_text(json.dumps({"summary": summary, "details": details}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
