#!/usr/bin/env python
"""Make a maximally strict/minimal tool-calling SFT file for preprocessing.

Azure/OpenAI tool-call fine-tune preprocessing may reject schemas unless they use
current strict tool schema rules:
  - function.strict = true
  - parameters.type = object
  - additionalProperties = false
  - required contains every property
  - no defaults, no unsupported annotations
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SRC_DIR = Path("artifacts/dev3-teacher-gold-small")
OUT_DIR = Path("artifacts/dev3-teacher-gold-small-strict")
VALID_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


def clean_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "string"}
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t in VALID_TYPES), "string")
    if schema_type not in VALID_TYPES:
        schema_type = "string"

    out: dict[str, Any] = {"type": schema_type}
    if isinstance(schema.get("enum"), list):
        enum = [x for x in schema["enum"] if x not in (None, "")]
        if enum:
            out["enum"] = enum
    if schema_type == "object":
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        out["properties"] = {str(k): clean_schema(v) for k, v in props.items()}
        out["required"] = list(out["properties"].keys())
        out["additionalProperties"] = False
    elif schema_type == "array":
        out["items"] = clean_schema(schema.get("items") or {"type": "string"})
    return out


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    fn = dict(tool.get("function", {}))
    params = clean_schema(fn.get("parameters") or {"type": "object", "properties": {}})
    if params.get("type") != "object":
        params = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description") or f"Call {fn['name']}.",
            "parameters": params,
            "strict": True,
        },
    }


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for msg in record["messages"]:
        role = msg.get("role")
        if role == "assistant":
            messages.append({
                "role": "assistant",
                "tool_calls": msg.get("tool_calls", []),
            })
        else:
            messages.append({"role": role, "content": msg.get("content", "")})
    return {"messages": messages, "tools": [normalize_tool(t) for t in record.get("tools", [])]}


def convert(name: str) -> None:
    rows = [json.loads(line) for line in (SRC_DIR / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = [normalize_record(row) for row in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized), encoding="utf-8")
    print(f"{name}: {len(normalized)} rows -> {OUT_DIR / name}")


if __name__ == "__main__":
    convert("train.jsonl")
    convert("eval.jsonl")
