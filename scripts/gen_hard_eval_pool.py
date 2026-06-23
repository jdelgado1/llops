#!/usr/bin/env python
"""
Option B: build a HARD eval pool from the toughest BFCL categories.

gpt-4.1-nano is strong on simple/multiple/parallel tool-calling (~88% on clean
items), so the teacher>>student gap is small. The parallel_multiple / live
categories require selecting AND parameterizing several functions at once —
where small models reliably trail frontier models.

Writes artifacts/eval_pool_hard.jsonl (messages, tools, reference_tool_calls),
the same shape the eval harness expects. Schemas are already sanitized by the
HF loader.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.tooldata import load_toolcalling

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Hardest AST-checkable single-turn categories (multi_turn needs stateful eval).
HARD_CATEGORIES = {"parallel_multiple", "live_parallel_multiple", "parallel", "live_parallel"}
OUT = ROOT / "artifacts" / "eval_pool_hard.jsonl"
LIMIT = 120


def main() -> None:
    logger.info(f"Pulling BFCL hard categories from HF: {sorted(HARD_CATEGORIES)}")
    # eval_pct=100 -> put everything in the eval pool (we want a pure hard eval set).
    _, eval_pool = load_toolcalling(
        source="hf", eval_pct=100, limit=LIMIT, categories=HARD_CATEGORIES
    )
    logger.info(f"Loaded {len(eval_pool)} hard items")
    logger.info(f"Category mix: {dict(Counter(it.category for it in eval_pool))}")

    records = []
    for it in eval_pool:
        records.append({
            "messages": it.messages,
            "tools": it.tools,
            "reference_tool_calls": it.reference,
            "category": it.category,
        })

    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    logger.info(f"✅ Wrote {len(records)} hard eval items -> {OUT}")
    logger.info("Next: quick teacher(gpt-5.4) vs nano check to confirm the gap.")


if __name__ == "__main__":
    main()
