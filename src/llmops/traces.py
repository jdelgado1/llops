"""Generate teacher traces for distillation.

Runs the grounded teacher (Web Search / WebIQ) over the **trace-generation pool**
(disjoint from the held-out eval pool) and records `(question, grounded answer,
citations)` as JSONL. These traces are the ONLY data the student is distilled on.

Design choice (v1): each example is **question -> grounded answer** (agent-style
distillation). The teacher's answer already carries inline citations, so the
student learns the domain synthesis + house style + sourced format. A
synthesis-over-frozen-context variant is possible later; we keep v1 simple.

Note: this calls live web search per question, so it costs money/time — start
small with ``--limit``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import get_settings
from .data import load_retrievalqa
from .teacher import GroundedTeacher

ARTIFACTS = Path("artifacts")


def generate_traces(limit: int | None = None, eval_pct: int = 30, out_path: str | None = None) -> Path:
    settings = get_settings()
    trace_pool, _ = load_retrievalqa(eval_pct=eval_pct)
    if limit is not None:
        trace_pool = trace_pool[:limit]
    n = len(trace_pool)
    if n == 0:
        raise RuntimeError("Trace pool is empty — check the dataset/split settings.")

    ARTIFACTS.mkdir(exist_ok=True)
    out = Path(out_path) if out_path else ARTIFACTS / f"traces-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    print(f"Generating {n} grounded teacher traces ({settings.teacher_model}, Web Search/WebIQ)")
    print(f"  -> {out}\n")

    written = 0
    with GroundedTeacher(settings) as teacher, out.open("w", encoding="utf-8") as f:
        for i, item in enumerate(trace_pool, 1):
            try:
                ans = teacher.ask(item.question)
            except Exception as exc:  # keep going; one bad question shouldn't kill the run
                print(f"[{i}/{n}] SKIP ({exc.__class__.__name__})  {item.question[:55]}")
                continue
            record = {
                "qid": item.qid,
                "question": item.question,
                "answer": ans.answer,
                "citations": ans.citations,
                "reference": item.answers,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{i}/{n}] OK ({len(ans.citations)} cites)  {item.question[:55]}")

    print(f"\nWrote {written}/{n} traces -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate grounded teacher traces for distillation.")
    ap.add_argument("--limit", type=int, default=None, help="cap number of trace-pool questions")
    ap.add_argument("--eval-pct", type=int, default=30, help="held-out eval percentage (rest is trace pool)")
    ap.add_argument("--out", type=str, default=None, help="output JSONL path")
    args = ap.parse_args()
    generate_traces(limit=args.limit, eval_pct=args.eval_pct, out_path=args.out)


if __name__ == "__main__":
    main()
