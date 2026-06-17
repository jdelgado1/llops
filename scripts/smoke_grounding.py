"""Smoke test: prove the grounded teacher (Web Search / WebIQ) works end-to-end.

Run after `pip install -r requirements.txt` and `az login`:

    python scripts/smoke_grounding.py

If this prints a current-events answer with citation URLs, your Foundry
project + Web Search tool are wired correctly and we can build the loop on top.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.config import get_settings
from llmops.teacher import GroundedTeacher

QUESTION = (
    "What are the most significant telecom, streaming, or gaming industry "
    "announcements from the past week? Give a short briefing with sources."
)


def main() -> None:
    settings = get_settings()
    print(f"Project : {settings.project_endpoint}")
    print(f"Teacher : {settings.teacher_model} (Web Search / WebIQ)\n")
    print(f"Q: {QUESTION}\n")

    with GroundedTeacher(settings) as teacher:
        ans = teacher.ask(QUESTION)

    print("A:", ans.answer, "\n")
    if ans.citations:
        print("Citations:")
        for url in ans.citations:
            print(f"  - {url}")
    else:
        print("(no citations returned — check instructions / tool_choice)")


if __name__ == "__main__":
    main()
