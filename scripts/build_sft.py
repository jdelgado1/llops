"""Build an OpenAI-format SFT dataset from teacher traces.

    python scripts/build_sft.py --traces artifacts/traces-XXNN.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.sft_dataset import main

if __name__ == "__main__":
    main()
