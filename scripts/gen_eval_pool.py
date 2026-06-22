#!/usr/bin/env python
"""
Generate eval pool from BFCL data (20% held-out, no training leakage).

Outputs ToolCallItem dicts in Foundry SFT format with reference_tool_calls added.
"""

import json
import logging
from pathlib import Path
from llmops.tooldata import load_toolcalling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load full dataset (300 items, 80% train / 20% eval)
    logger.info("Loading BFCL data...")
    trace_pool, eval_pool = load_toolcalling(source="hf", eval_pct=20, limit=300)

    logger.info(f"Train pool: {len(trace_pool)} items")
    logger.info(f"Eval pool: {len(eval_pool)} items")

    # Convert eval pool to Foundry SFT format with reference
    eval_records = []
    for item in eval_pool:
        # Each item is a ToolCallItem with:
        # - messages: [system, user, assistant]
        # - tools: list of tool schemas
        # - reference: the reference/correct calls

        record = {
            "messages": item.messages,
            "tools": item.tools,
            "reference_tool_calls": item.reference,  # Ground truth
        }
        eval_records.append(record)

    # Write JSONL
    output_path = Path("artifacts/eval_pool_114items.jsonl")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in eval_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"✅ Wrote {len(eval_records)} eval items to {output_path}")
    logger.info(f"   Ready for: python scripts/eval_tool_with_fabric_export.py --eval-pool-jsonl {output_path}")


if __name__ == "__main__":
    main()
