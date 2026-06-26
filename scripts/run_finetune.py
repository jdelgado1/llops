"""Submit a Foundry managed SFT job, or check a job's status.

    # kick off a fine-tune
    python scripts/run_finetune.py submit --training-file data/format_primer.jsonl \
        --model Qwen3-32B --suffix format-primer --epochs 4

    # check status later
    python scripts/run_finetune.py status --job-id ftjob-XXXX --watch
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.finetune import main

if __name__ == "__main__":
    main()
