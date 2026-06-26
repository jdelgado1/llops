"""Build a function-calling SFT dataset from teacher tool-call traces.

    python scripts/build_tool_sft.py --traces artifacts/tool-traces-XXNN.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.tool_sft import main

if __name__ == "__main__":
    main()
