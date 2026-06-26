"""Generate teacher tool-call traces for distillation (rejection-sampled).

Runs the frontier teacher (GPT-5.4) over the **trace-generation pool** (disjoint
from the held-out eval pool), captures its tool call(s), and **keeps only the
ones that are AST-correct** against the reference. These rejection-sampled,
correct traces are the ONLY data the student is distilled on — which is exactly
what lets a small student match/beat the teacher.

In the live demo these traces come from Foundry Tracing of the hosted agent;
here we generate them by calling the teacher directly. Each kept record stores
the messages, the offered tools, and the teacher's correct call(s).

Note: this calls the teacher per item, so start small with ``--limit``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .ast_check import check_ast
from .config import get_settings
from .models import get_client
from .tool_models import call_with_tools
from .tooldata import load_toolcalling

ARTIFACTS = Path("artifacts")


def generate_traces(
    source: str | None = None,
    limit: int | None = None,
    eval_pct: int = 30,
    out_path: str | None = None,
) -> Path:
    settings = get_settings()
    source = source or settings.toolcalling_source
    trace_pool, _ = load_toolcalling(source=source, eval_pct=eval_pct)
    if limit is not None:
        trace_pool = trace_pool[:limit]
    n = len(trace_pool)
    if n == 0:
        raise RuntimeError("Trace pool is empty — check the source/split settings.")

    client = get_client(settings)
    ARTIFACTS.mkdir(exist_ok=True)
    out = Path(out_path) if out_path else ARTIFACTS / f"tool-traces-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    print(f"Generating teacher tool-call traces ({settings.teacher_model}) over {n} prompts")
    print(f"  source={source}  ->  {out}\n")

    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for i, item in enumerate(trace_pool, 1):
            try:
                calls, _usage, _lat = call_with_tools(
                    client, settings.teacher_model, item.messages, item.tools,
                    tool_choice="required",
                )
            except Exception as exc:  # one bad item shouldn't kill the run
                print(f"[{i}/{n}] SKIP ({exc.__class__.__name__})  {item.user_text()[:50]}")
                continue

            verdict = check_ast(calls, item.reference, item.tools)
            if not verdict.correct:
                print(f"[{i}/{n}] REJECT ({verdict.reason})  {item.user_text()[:46]}")
                continue

            record = {
                "tid": item.tid,
                "category": item.category,
                "messages": item.messages,
                "tools": item.tools,
                "tool_calls": calls,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            print(f"[{i}/{n}] KEEP  {item.user_text()[:52]}")

    print(f"\nKept {kept}/{n} AST-correct teacher traces -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate rejection-sampled teacher tool-call traces.")
    ap.add_argument("--source", default=None, help="sample | hf | path (default: env TOOLCALLING_SOURCE)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of trace-pool prompts")
    ap.add_argument("--eval-pct", type=int, default=30, help="held-out eval percentage (rest is trace pool)")
    ap.add_argument("--out", default=None, help="output JSONL path")
    args = ap.parse_args()
    generate_traces(source=args.source, limit=args.limit, eval_pct=args.eval_pct, out_path=args.out)


if __name__ == "__main__":
    main()
