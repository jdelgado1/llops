"""Run the 3-way tool-calling AST eval (teacher / baseline / distilled).

    python scripts/run_tool_eval.py --limit 20
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.tool_eval import main

if __name__ == "__main__":
    main()
