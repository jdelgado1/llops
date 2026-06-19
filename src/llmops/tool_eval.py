"""The headline eval: 3-way **AST accuracy** + cost/latency.

Scores three deployments on the held-out tool-calling eval pool (disjoint from
the traces the student trained on), with the single objective metric:

  - teacher   = gpt-5.4         (frontier quality bar)
  - baseline  = OPTIONAL "before" proxy (base Qwen3-32B inference is unavailable
                on Foundry, so use an off-task / format-primer fine-tune, or omit)
  - distilled = SFT'd Qwen3-32B (the "after" — the result)

Target headline: ``distilled >= teacher`` (and ``>> baseline`` when present). We also record latency
(p50/p95) and token usage per model — the "parity at lower cost" number that
keeps quality (AST %) and performance (latency/tokens) as *separate* numbers.

Run:
    python scripts/run_tool_eval.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

from .ast_check import check_ast
from .config import get_settings
from .models import get_client
from .tool_models import call_with_tools
from .tooldata import load_toolcalling

ARTIFACTS = Path("artifacts")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100) * (len(s) - 1)))))
    return round(s[k], 3)


def _tokens(usage) -> int:
    return int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0


def run(
    source: str | None = None,
    limit: int | None = None,
    eval_pct: int = 30,
) -> dict:
    settings = get_settings()
    source = source or settings.toolcalling_source
    client = get_client(settings)

    # Base Qwen3-32B inference is NOT available on Foundry (fine-tune-only), so the
    # "baseline" column is OPTIONAL: supply an off-task / format-primer proxy if you
    # have one, otherwise the headline stands on teacher vs distilled alone.
    candidate = {
        "teacher": settings.teacher_model,
        "baseline": settings.baseline_deployment,
        "distilled": settings.student_finetuned_deployment,
    }
    models: dict[str, str] = {k: v for k, v in candidate.items() if v}
    required = [k for k in ("teacher", "distilled") if k not in models]
    if required:
        raise RuntimeError(f"Missing required deployment names for: {required} (check .env)")

    _, eval_pool = load_toolcalling(source=source, eval_pct=eval_pct)
    if limit is not None:
        eval_pool = eval_pool[:limit]
    n = len(eval_pool)
    if n == 0:
        raise RuntimeError("Eval pool is empty.")

    print(f"Source: {source} | Eval pool: {n} held-out tool-calling items | Metric: AST accuracy")
    for name, dep in models.items():
        print(f"  {name:9s} -> {dep}")
    print()

    correct = {name: 0 for name in models}
    latencies: dict[str, list[float]] = {name: [] for name in models}
    out_tokens: dict[str, list[int]] = {name: [] for name in models}
    rows = []

    for i, item in enumerate(eval_pool, 1):
        row = {"tid": item.tid, "category": item.category, "request": item.user_text()}
        marks = []
        for name, dep in models.items():
            try:
                calls, usage, latency_s = call_with_tools(client, dep, item.messages, item.tools)
            except Exception as exc:
                calls, usage, latency_s = [], None, 0.0
                row[f"{name}_error"] = f"{exc.__class__.__name__}: {exc}"
            verdict = check_ast(calls, item.reference, item.tools)
            correct[name] += int(verdict.correct)
            latencies[name].append(latency_s)
            out_tokens[name].append(_tokens(usage))
            row[f"{name}_calls"] = calls
            row[f"{name}_correct"] = verdict.correct
            marks.append(f"{name}={'OK ' if verdict.correct else 'MISS'}")
        rows.append(row)
        print(f"[{i}/{n}] " + "  ".join(marks) + f"  {item.user_text()[:40]}")

    ast = {name: round(100 * c / n, 1) for name, c in correct.items()}
    perf = {
        name: {
            "p50_latency_s": _percentile(latencies[name], 50),
            "p95_latency_s": _percentile(latencies[name], 95),
            "avg_out_tokens": round(sum(out_tokens[name]) / n, 1),
        }
        for name in models
    }
    summary = {
        "source": source,
        "metric": "ast_accuracy",
        "n": n,
        "deployments": models,
        "ast_accuracy": ast,
        "performance": perf,
        "delta_distilled_vs_teacher": round(ast["distilled"] - ast["teacher"], 1),
    }
    if "baseline" in ast:
        summary["lift_distilled_vs_baseline"] = round(ast["distilled"] - ast["baseline"], 1)

    print("\n=== AST accuracy (the scoreboard) ===")
    print(f"  teacher   (gpt-5.4)        : {ast['teacher']}%")
    if "baseline" in ast:
        print(f"  baseline  (qwen proxy)     : {ast['baseline']}%")
    print(f"  distilled (qwen, after)    : {ast['distilled']}%")
    if "baseline" in ast:
        print(f"  --> distillation lift (after - before) : {summary['lift_distilled_vs_baseline']:+} pts")
    print(f"  --> distilled vs teacher               : {summary['delta_distilled_vs_teacher']:+} pts")
    print("\n=== Performance (separate from quality) ===")
    for name in models:
        p = perf[name]
        print(f"  {name:9s}  p50={p['p50_latency_s']}s  p95={p['p95_latency_s']}s  out_tok={p['avg_out_tokens']}")

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / f"tool-eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="3-way tool-calling AST eval (teacher / baseline / distilled).")
    ap.add_argument("--source", default=None, help="sample | hf | path (default: env TOOLCALLING_SOURCE)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-pct", type=int, default=30)
    args = ap.parse_args()
    run(source=args.source, limit=args.limit, eval_pct=args.eval_pct)


if __name__ == "__main__":
    main()
