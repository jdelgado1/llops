"""Baseline eval: frontier teacher vs base student, scored by the Answer Quality Score.

This is the "baseline first" step (and the demo's opening punchline): show the
quality GAP between the expensive teacher (GPT-5.4) and the cheap base student
(gpt-4.1-mini) on the held-out RetrievalQA eval pool — using FROZEN context (the
reproducible regression leg of the hybrid; no live web-search calls).

Distillation later closes this gap. Run:

    python scripts/run_baseline.py --limit 20
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

from .config import get_settings
from .data import load_retrievalqa, load_webbench
from .judge import judge_answer
from .models import answer_closed_book, answer_over_context, get_client
from .teacher import GroundedTeacher

ARTIFACTS = Path("artifacts")

MODES = ("frozen_context", "closed_book", "web_search")
DATASETS = ("retrievalqa", "webbench")


def run(
    limit: int | None = None,
    eval_pct: int = 30,
    mode: str = "frozen_context",
    dataset: str = "retrievalqa",
    split: str = "webwalkerqa_ref",
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    settings = get_settings()
    client = get_client(settings)

    if dataset == "webbench":
        eval_pool = load_webbench(split=split, limit=limit)
    else:
        _, eval_pool = load_retrievalqa(eval_pct=eval_pct)
        if limit is not None:
            eval_pool = eval_pool[:limit]
    n = len(eval_pool)
    if n == 0:
        raise RuntimeError("Eval pool is empty — check the dataset/split settings.")

    label = f"{dataset}:{split}" if dataset == "webbench" else dataset
    print(f"Dataset:   {label}")
    print(f"Mode:      {mode}")
    print(f"Eval pool: {n} held-out questions")
    print(f"Teacher:   {settings.teacher_model}")
    print(f"Student:   {settings.student_model}")
    print(f"Judge:     {settings.judge_model}\n")

    rows = []
    teacher_correct = 0
    student_correct = 0

    with contextlib.ExitStack() as stack:
        # Pick how each model produces an answer for this mode.
        if mode == "web_search":
            t_agent = stack.enter_context(GroundedTeacher(settings, model=settings.teacher_model))
            s_agent = stack.enter_context(GroundedTeacher(settings, model=settings.student_model))

            def teacher_answer(q: str, ctx: str) -> str:
                return t_agent.ask(q).answer

            def student_answer(q: str, ctx: str) -> str:
                return s_agent.ask(q).answer
        elif mode == "closed_book":
            def teacher_answer(q: str, ctx: str) -> str:
                return answer_closed_book(client, settings.teacher_model, q)

            def student_answer(q: str, ctx: str) -> str:
                return answer_closed_book(client, settings.student_model, q)
        else:  # frozen_context
            def teacher_answer(q: str, ctx: str) -> str:
                return answer_over_context(client, settings.teacher_model, q, ctx)

            def student_answer(q: str, ctx: str) -> str:
                return answer_over_context(client, settings.student_model, q, ctx)

        for i, item in enumerate(eval_pool, 1):
            context = item.context_text()
            teacher_ans = teacher_answer(item.question, context)
            student_ans = student_answer(item.question, context)
            t_j = judge_answer(client, settings.judge_model, item.question, item.answers, teacher_ans)
            s_j = judge_answer(client, settings.judge_model, item.question, item.answers, student_ans)
            teacher_correct += int(t_j.correct)
            student_correct += int(s_j.correct)
            rows.append(
                {
                    "qid": item.qid,
                    "question": item.question,
                    "reference": item.answers,
                    "teacher_answer": teacher_ans,
                    "student_answer": student_ans,
                    "teacher_correct": t_j.correct,
                    "student_correct": s_j.correct,
                    "teacher_reason": t_j.reason,
                    "student_reason": s_j.reason,
                }
            )
            print(
                f"[{i}/{n}] teacher={'OK ' if t_j.correct else 'MISS'} "
                f"student={'OK ' if s_j.correct else 'MISS'}  {item.question[:60]}"
            )

    teacher_score = round(100 * teacher_correct / n, 1)
    student_score = round(100 * student_correct / n, 1)
    summary = {
        "dataset": label,
        "mode": mode,
        "n": n,
        "teacher_model": settings.teacher_model,
        "student_model": settings.student_model,
        "judge_model": settings.judge_model,
        "teacher_quality_score": teacher_score,
        "student_quality_score": student_score,
        "gap": round(teacher_score - student_score, 1),
    }

    print(f"\n=== Answer Quality Score ({mode}) ===")
    print(f"  Teacher ({settings.teacher_model}): {teacher_score}%")
    print(f"  Student ({settings.student_model}): {student_score}%")
    print(f"  Gap (teacher - student):            {summary['gap']} points")

    ARTIFACTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ARTIFACTS / f"baseline-{mode}-{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved per-question results -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline eval: teacher vs base student.")
    ap.add_argument("--limit", type=int, default=None, help="cap number of eval questions (cheap runs)")
    ap.add_argument("--eval-pct", type=int, default=30, help="held-out eval percentage")
    ap.add_argument(
        "--mode",
        choices=MODES,
        default="frozen_context",
        help="frozen_context (regression guardrail) | closed_book (floor) | web_search (the real gap)",
    )
    ap.add_argument(
        "--dataset",
        choices=DATASETS,
        default="retrievalqa",
        help="retrievalqa (easy factoid) | webbench (hard multi-hop research)",
    )
    ap.add_argument(
        "--split",
        default="webwalkerqa_ref",
        help="web-bench split: webwalkerqa_ref | seal_ref | gaia_text",
    )
    args = ap.parse_args()
    run(limit=args.limit, eval_pct=args.eval_pct, mode=args.mode, dataset=args.dataset, split=args.split)


if __name__ == "__main__":
    main()
