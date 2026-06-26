"""Smoke test: prove tool calling + AST scoring work end-to-end.

Run after `pip install -r requirements.txt` and `az login`:

    python scripts/smoke_toolcalling.py

It loads one bundled TMG item, asks the teacher to call a tool, parses the call,
and prints the AST verdict. If you see a correct call + "AST: CORRECT", the
Foundry endpoint, the tool interface, and the metric are all wired correctly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmops.ast_check import check_ast
from llmops.config import get_settings
from llmops.models import get_client
from llmops.tool_models import call_with_tools
from llmops.tooldata import load_toolcalling


def main() -> None:
    settings = get_settings()
    print(f"Project : {settings.project_endpoint}")
    print(f"Teacher : {settings.teacher_model}\n")

    trace_pool, eval_pool = load_toolcalling(source=settings.toolcalling_source)
    item = (eval_pool or trace_pool)[0]
    print(f"Request : {item.user_text()}")
    print(f"Tools   : {[t['function']['name'] for t in item.tools]}\n")

    client = get_client(settings)
    calls, usage, latency_s = call_with_tools(
        client, settings.teacher_model, item.messages, item.tools, tool_choice="required"
    )
    print("Predicted call(s):")
    print(json.dumps(calls, indent=2))

    verdict = check_ast(calls, item.reference, item.tools)
    print(f"\nAST: {'CORRECT' if verdict.correct else 'WRONG'} ({verdict.reason})")
    print(f"Latency: {latency_s:.2f}s")


if __name__ == "__main__":
    main()
