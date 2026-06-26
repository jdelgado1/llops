#!/usr/bin/env python
"""Benchmark latency/RPS for GPT-5.4 vs GPT-4.1-nano on sampled prompts.

Samples prompts from a JSONL dataset (default: artifacts/eval_pool_hard.jsonl),
then sends the same tool-calling request to each deployment.

Reports per deployment:
  - request count
  - success/error count
  - average end-to-end latency
  - p50/p95 latency
  - wall-clock duration
  - requests per second (RPS)
  - token totals when usage is returned

Example:
  python scripts/benchmark_deployments_latency.py --sample-size 100 --concurrency 5
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.config import get_settings
from llmops.models import get_client
from llmops.tool_models import _messages_with_system

DEFAULT_DATASET = ROOT / "artifacts" / "eval_pool_hard.jsonl"
OUT_DIR = ROOT / "artifacts" / "benchmarks"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = idx - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_sample(path: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < sample_size:
        raise ValueError(f"dataset has {len(rows)} rows but sample-size={sample_size}")
    rng = random.Random(seed)
    indexed = list(enumerate(rows))
    return [{"sample_index": idx, **row} for idx, row in rng.sample(indexed, sample_size)]


def invoke_one(client, deployment: str, sample: dict[str, Any], timeout_label: str) -> dict[str, Any]:
    messages = _messages_with_system(sample.get("messages", []))
    tools = sample.get("tools", [])
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=512,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens)
        return {
            "deployment": deployment,
            "sample_index": sample["sample_index"],
            "ok": True,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "error": None,
        }
    except Exception as exc:  # keep benchmark moving across throttles/schema issues
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "deployment": deployment,
            "sample_index": sample["sample_index"],
            "ok": False,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }


def benchmark_deployment(client, deployment: str, samples: list[dict[str, Any]], concurrency: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke_one, client, deployment, sample, deployment) for sample in samples]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % 10 == 0 or completed == len(samples):
                print(f"{deployment}: {completed}/{len(samples)}", flush=True)
    wall_s = time.perf_counter() - started
    ok_results = [r for r in results if r["ok"]]
    latencies = [r["latency_ms"] for r in ok_results]
    summary = {
        "deployment": deployment,
        "request_count": len(results),
        "success_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "wall_clock_seconds": wall_s,
        "rps_all_requests": len(results) / wall_s if wall_s else 0.0,
        "rps_successful_requests": len(ok_results) / wall_s if wall_s else 0.0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "prompt_tokens": sum(r["prompt_tokens"] for r in ok_results),
        "completion_tokens": sum(r["completion_tokens"] for r in ok_results),
        "total_tokens": sum(r["total_tokens"] for r in ok_results),
    }
    return summary, sorted(results, key=lambda r: r["sample_index"])


def write_outputs(run_id: str, summaries: list[dict[str, Any]], details: list[dict[str, Any]], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "summaries": summaries,
    }
    (OUT_DIR / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT_DIR / f"{run_id}_details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["deployment", "sample_index", "ok", "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens", "error"])
        writer.writeheader()
        writer.writerows(details)
    print(f"Wrote {OUT_DIR / f'{run_id}.json'}")
    print(f"Wrote {OUT_DIR / f'{run_id}_details.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--teacher-deployment", default="gpt-5.4")
    parser.add_argument("--nano-deployment", default="gpt-41-nano-base")
    args = parser.parse_args()

    samples = load_sample(args.dataset, args.sample_size, args.seed)
    client = get_client(get_settings())
    run_id = f"latency_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    all_details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for deployment in [args.teacher_deployment, args.nano_deployment]:
        print(f"\nBenchmarking {deployment} on {len(samples)} prompts (concurrency={args.concurrency})", flush=True)
        summary, details = benchmark_deployment(client, deployment, samples, args.concurrency)
        summaries.append(summary)
        all_details.extend(details)
    write_outputs(run_id, summaries, all_details, args)
    print("\nSUMMARY")
    for summary in summaries:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
