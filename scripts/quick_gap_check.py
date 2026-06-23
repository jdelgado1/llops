#!/usr/bin/env python
"""Quick teacher(gpt-5.4) vs nano gap check on a sample of the HARD eval pool."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.models import invoke_model
from llmops.ast_check import check_ast

logging.basicConfig(level=logging.WARNING)

POOL = ROOT / "artifacts" / "eval_pool_hard.jsonl"
NANO = "gpt-41-nano-base"
TEACHER = "gpt-5.4"
N = 30


def score(deployment: str, items: list[dict]) -> tuple[int, int]:
    ok = 0
    for it in items:
        try:
            res = invoke_model(deployment=deployment, messages=it.get("messages", []),
                               tools=it.get("tools", []))
            r = check_ast(pred_calls=res.get("tool_calls", []),
                          ref_calls=it.get("reference_tool_calls", []),
                          tools=it.get("tools", []))
            if r.correct:
                ok += 1
        except Exception as e:
            print(f"  {deployment} err: {str(e)[:80]}")
    return ok, len(items)


def main() -> None:
    items = [json.loads(l) for l in POOL.read_text(encoding="utf-8").splitlines() if l.strip()][:N]
    print(f"Hard-pool gap check on {len(items)} items ({POOL.name})\n")

    print(f"Running nano ({NANO}) ...")
    n_ok, n = score(NANO, items)
    print(f"Running teacher ({TEACHER}) ...")
    t_ok, _ = score(TEACHER, items)

    nano_pct = n_ok / n * 100
    teach_pct = t_ok / n * 100
    print("\n=== HARD-POOL GAP ===")
    print(f"  nano    : {nano_pct:5.1f}%  ({n_ok}/{n})")
    print(f"  teacher : {teach_pct:5.1f}%  ({t_ok}/{n})")
    print(f"  GAP     : {teach_pct - nano_pct:5.1f} pts")
    if teach_pct - nano_pct >= 20:
        print("  ✅ Big gap — great distillation demo material.")
    elif teach_pct - nano_pct >= 10:
        print("  ◑ Moderate gap.")
    else:
        print("  ⚠️  Small gap — nano handles these too; try parallel_multiple / multi_turn.")


if __name__ == "__main__":
    main()
