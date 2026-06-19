"""Distillation eval: distilled (TMG-tuned) vs baseline (GSM8K-tuned) vs teacher.

Scores three models on the held-out RetrievalQA eval pool using FROZEN context
(reproducible), with the single Answer Quality Score. This is the "before vs
after vs frontier" view:

  - teacher        = gpt-5.4               (the quality bar)
  - baseline_gsm8k = qwen3-32b tuned on math (proxy for untuned-on-our-task)
  - distilled      = qwen3-32b tuned on OUR traces (the result)

Run:
    python scripts/run_distill_eval.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import get_settings
from .data import load_retrievalqa, load_webbench
from .judge import judge_answer, score_match
from .models import answer_closed_book, answer_over_context, get_client

ARTIFACTS = Path("artifacts")
MODES = ("frozen_context", "closed_book")
DATASETS = ("retrievalqa", "webbench")
SCORERS = ("match", "judge")


def run(
    limit: int | None = None,
    eval_pct: int = 30,
    mode: str = "frozen_context",
    dataset: str = "retrievalqa",
    split: str = "webwalkerqa_ref",
    scorer: str = "match",
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {SCORERS}")
    if dataset == "webbench" and mode == "frozen_context":
        raise ValueError("web-bench has no bundled context; use --mode closed_book")

    settings = get_settings()
    client = get_client(settings)

    models: dict[str, str | None] = {
        "teacher": settings.teacher_model,
        "baseline_gsm8k": settings.baseline_deployment,
        "distilled": settings.student_finetuned_deployment,
    }
    missing = [k for k, v in models.items() if not v]
    if missing:
        raise RuntimeError(f"Missing deployment names for: {missing} (check .env)")

    if dataset == "webbench":
        eval_pool = load_webbench(split=split, limit=limit)
        label = f"webbench:{split}"
    else:
        _, eval_pool = load_retrievalqa(eval_pct=eval_pct)
        if limit is not None:
            eval_pool = eval_pool[:limit]
        label = "retrievalqa"
    n = len(eval_pool)
    if n == 0:
        raise RuntimeError("Eval pool is empty.")

    def answer(dep: str, item) -> str:
        if mode == "closed_book":
            return answer_closed_book(client, dep, item.question)
        return answer_over_context(client, dep, item.question, item.context_text())

    def grade(refs, cand):
        if scorer == "judge":
            return judge_answer(client, settings.judge_model, "", refs, cand)
        return score_match(refs, cand)

    print(f"Dataset: {label} | Mode: {mode} | Scorer: {scorer}")
    for name, dep in models.items():
        print(f"  {name:14s} -> {dep}")
    print(f"Eval pool: {n} held-out questions\n")

    correct = {name: 0 for name in models}
    rows = []
    for i, item in enumerate(eval_pool, 1):
        row = {"qid": item.qid, "question": item.question, "reference": item.answers}
        marks = []
        for name, dep in models.items():
            ans = answer(dep, item)
            j = grade(item.answers, ans)
            correct[name] += int(j.correct)
            row[f"{name}_answer"] = ans
            row[f"{name}_correct"] = j.correct
            marks.append(f"{name}={'OK ' if j.correct else 'MISS'}")
        rows.append(row)
        print(f"[{i}/{n}] " + "  ".join(marks) + f"  {item.question[:42]}")

    scores = {name: round(100 * c / n, 1) for name, c in correct.items()}
    summary = {
        "dataset": label,
        "mode": mode,
        "scorer": scorer,
        "n": n,
        "deployments": models,
        "scores": scores,
        "lift_distilled_vs_baseline": round(scores["distilled"] - scores["baseline_gsm8k"], 1),
        "gap_distilled_vs_teacher": round(scores["teacher"] - scores["distilled"], 1),
    }

    print(f"\n=== Answer Quality Score ({label}, {mode}, {scorer}) ===")
    print(f"  teacher        (gpt-5.4)          : {scores['teacher']}%")
    print(f"  baseline_gsm8k (qwen, before)     : {scores['baseline_gsm8k']}%")
    print(f"  distilled      (qwen, after)      : {scores['distilled']}%")
    print(f"  --> distillation lift (after-before): {summary['lift_distilled_vs_baseline']:+} pts")
    print(f"  --> remaining gap to teacher        : {summary['gap_distilled_vs_teacher']} pts")

    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / f"distill-{label.replace(':', '-')}-{mode}-{scorer}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Distilled vs baseline vs teacher eval.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-pct", type=int, default=30)
    ap.add_argument("--mode", choices=MODES, default="frozen_context")
    ap.add_argument("--dataset", choices=DATASETS, default="retrievalqa")
    ap.add_argument("--split", default="webwalkerqa_ref", help="web-bench split")
    ap.add_argument("--scorer", choices=SCORERS, default="match", help="match (objective) | judge (LLM)")
    args = ap.parse_args()
    run(
        limit=args.limit,
        eval_pct=args.eval_pct,
        mode=args.mode,
        dataset=args.dataset,
        split=args.split,
        scorer=args.scorer,
    )


if __name__ == "__main__":
    main()
