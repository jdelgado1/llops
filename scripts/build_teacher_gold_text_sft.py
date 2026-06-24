#!/usr/bin/env python
"""Build schema-free text tool-call SFT files.

Azure fine-tune preprocessing is rejecting native tool schemas. The runtime parser
already supports <tool_call>{...}</tool_call> text, so this format avoids all
schema validation while preserving tool-call behavior for eval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SRC_DIR = Path("artifacts/dev3-teacher-gold-small")
OUT_DIR = Path("artifacts/dev3-teacher-gold-small-text")

SYSTEM = (
    "You are a TMG operations tool-calling agent. For every request, output only "
    "one or more tool calls using this exact format, with no prose: "
    "<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>"
)


def args_from_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc["function"]
    args = json.loads(fn.get("arguments") or "{}")
    return {"name": fn["name"], "arguments": args}


def convert_record(row: dict[str, Any]) -> dict[str, Any]:
    user = next(m for m in row["messages"] if m.get("role") == "user")
    assistant = next(m for m in row["messages"] if m.get("role") == "assistant")
    calls = [args_from_call(tc) for tc in assistant.get("tool_calls", [])]
    content = "\n".join(
        f"<tool_call>{json.dumps(call, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
        for call in calls
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user["content"]},
            {"role": "assistant", "content": content},
        ]
    }


def convert(name: str) -> None:
    rows = [json.loads(line) for line in (SRC_DIR / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    converted = [convert_record(row) for row in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
    print(f"{name}: {len(converted)} rows -> {OUT_DIR / name}")


if __name__ == "__main__":
    convert("train.jsonl")
    convert("eval.jsonl")
