"""Run the distillation eval (distilled vs GSM8K-baseline vs teacher).

    python scripts/run_distill_eval.py --limit 20
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.distill_eval import main

if __name__ == "__main__":
    main()
