"""Generate rejection-sampled teacher tool-call traces.

    python scripts/gen_tool_traces.py --limit 50
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.tool_traces import main

if __name__ == "__main__":
    main()
