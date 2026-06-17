"""Datasets and the by-question train/eval split.

HARD RULE (no test-set leakage): we never train on eval rows. RetrievalQA is
split *by question* into:
  - a trace-generation pool  -> teacher answers these -> becomes training data
  - a held-out eval pool      -> only ever used to score / gate

The split is deterministic (hash of the question id), so it's stable across
runs and reproducible for anyone re-running the demo.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from datasets import load_dataset

RETRIEVALQA = "aialt/RetrievalQA"


@dataclass
class QAItem:
    """One question with its reference answer(s) and (optional) frozen context."""

    qid: str
    question: str
    answers: list[str]
    context: list[dict] = field(default_factory=list)  # [{"title": str, "text": str}]
    needs_retrieval: bool = True

    def context_text(self) -> str:
        """Flatten the frozen retrieved context into a single prompt-ready block."""
        parts = []
        for i, c in enumerate(self.context, 1):
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
            header = f"[{i}] {title}".rstrip()
            parts.append(f"{header}\n{text}" if text else header)
        return "\n\n".join(parts)


def _eval_bucket(qid: str, buckets: int = 100) -> int:
    """Stable 0..buckets-1 bucket for a question id (deterministic split)."""
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def load_retrievalqa(
    eval_pct: int = 30,
    retrieval_only: bool = True,
    limit: int | None = None,
) -> tuple[list[QAItem], list[QAItem]]:
    """Load RetrievalQA and split into (trace_pool, eval_pool).

    Args:
        eval_pct: percent of questions held out for evaluation (never trained on).
        retrieval_only: keep only questions that *require* external retrieval
            (``param_knowledge_answerable == 0``) — these are the ones where
            grounding actually matters, which is the whole point of the demo.
        limit: optional cap on total questions (handy for quick smoke runs).
    """
    ds = load_dataset(RETRIEVALQA, split="train")

    items: list[QAItem] = []
    for row in ds:
        needs_retrieval = int(row.get("param_knowledge_answerable", 0) or 0) == 0
        if retrieval_only and not needs_retrieval:
            continue
        items.append(
            QAItem(
                qid=str(row.get("question_id")),
                question=row["question"],
                answers=list(row.get("ground_truth") or []),
                context=list(row.get("context") or []),
                needs_retrieval=needs_retrieval,
            )
        )
        if limit is not None and len(items) >= limit:
            break

    trace_pool: list[QAItem] = []
    eval_pool: list[QAItem] = []
    for it in items:
        if _eval_bucket(it.qid) < eval_pct:
            eval_pool.append(it)
        else:
            trace_pool.append(it)
    return trace_pool, eval_pool
