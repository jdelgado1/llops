#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.config import get_settings
from llmops.models import get_client
from llmops.tool_models import _parse_calls_from_text

EVAL = ROOT / "artifacts" / "dev3-teacher-gold-small-text" / "eval.jsonl"
DEPLOYMENTS = ["gpt-41-nano-base", "gpt-41-nano-student-v1"]

row = json.loads(EVAL.read_text(encoding="utf-8").splitlines()[0])
prompt = [m for m in row["messages"] if m.get("role") != "assistant"]
expected = next(m for m in row["messages"] if m.get("role") == "assistant")["content"]
client = get_client(get_settings())

print("USER:", prompt[-1]["content"])
print("EXPECTED:", expected)
for dep in DEPLOYMENTS:
    resp = client.chat.completions.create(model=dep, messages=prompt, max_completion_tokens=512)
    content = resp.choices[0].message.content or ""
    print("\nMODEL", dep)
    print("RAW:", content)
    print("PARSED:", _parse_calls_from_text(content))
