#!/usr/bin/env python
"""Normalize teacher-gold function-calling SFT JSONL for Azure OpenAI preprocessing.

The model preprocessor reported "contains invalid schema" for every line. This
rewrites the short teacher-gold train/eval files to a minimal, strict tool schema:
  - keep only messages + tools at top level
  - remove metadata
  - assistant content = null with tool_calls
  - tool parameters are strict JSON Schema objects
  - remove defaults and other loose schema annotations
  - additionalProperties=false for object schemas
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SRC_DIR = Path("artifacts/dev3-teacher-gold-small")
OUT_DIR = Path("artifacts/dev3-teacher-gold-small-normalized")
VALID_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


def clean_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "string"}

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t in VALID_TYPES and t != "null"), "string")
    if not isinstance(schema_type, str) or schema_type not in VALID_TYPES:
        schema_type = "string"

    out: dict[str, Any] = {"type": schema_type}
    if isinstance(schema.get("description"), str):
        out["description"] = schema["description"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        out["enum"] = [x for x in schema["enum"] if x not in (None, "")]

    if schema_type == "object":
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        out["properties"] = {str(k): clean_schema(v) for k, v in props.items()}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        out["required"] = [str(k) for k in required if str(k) in out["properties"]]
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
            "description": fn.get("description", ""),
            "parameters": params,
        },
    }


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for msg in record["messages"]:
        role = msg.get("role")
        if role == "assistant":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": msg.get("tool_calls", []),
            })
        else:
            messages.append({"role": role, "content": msg.get("content", "")})
    return {
        "messages": messages,
        "tools": [normalize_tool(t) for t in record.get("tools", [])],
    }


def convert(name: str) -> None:
    src = SRC_DIR / name
    out = OUT_DIR / name
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = [normalize_record(row) for row in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized), encoding="utf-8")
    print(f"{name}: {len(normalized)} rows -> {out}")


if __name__ == "__main__":
    convert("train.jsonl")
    convert("eval.jsonl")
