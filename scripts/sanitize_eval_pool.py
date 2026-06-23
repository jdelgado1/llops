#!/usr/bin/env python
"""
Sanitize the eval pool's tool schemas so they're valid for the OpenAI
function-calling API.

The pool was derived from a BFCL/ToolACE-style dataset that uses loose / Java
types ("String", "char", "any", ...) which the OpenAI API rejects with HTTP 400
(invalid_function_parameters). This normalizes those to valid JSON-Schema types.

Scope: only walks tools[].function.parameters (NOT the tool-level
{"type": "function"} marker, and NOT reference_tool_calls).

Backs up the original to <name>.raw.jsonl, then rewrites the canonical file.

Usage:
  python scripts/sanitize_eval_pool.py [path]   (default: artifacts/eval_pool_114items.jsonl)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

VALID = {"string", "number", "integer", "boolean", "array", "object", "null"}

TYPE_MAP = {
    # string-ish
    "str": "string", "string": "string", "char": "string", "character": "string",
    "text": "string", "uuid": "string", "date": "string", "datetime": "string",
    "time": "string", "timestamp": "string", "email": "string", "url": "string",
    "enum": "string",
    # integer-ish
    "int": "integer", "integer": "integer", "long": "integer", "short": "integer",
    "byte": "integer", "bigint": "integer",
    # number-ish
    "float": "number", "double": "number", "number": "number", "decimal": "number",
    "real": "number",
    # boolean
    "bool": "boolean", "boolean": "boolean",
    # array-ish
    "list": "array", "array": "array", "arraylist": "array", "tuple": "array",
    "set": "array", "collection": "array",
    # object-ish
    "dict": "object", "object": "object", "map": "object", "hashmap": "object",
    "json": "object", "hashtable": "object",
    # special
    "any": "string", "none": "null", "null": "null", "void": "null",
}

fixes = {"count": 0}


def norm_type(t: Any) -> Any:
    if not isinstance(t, str):
        return t
    key = t.strip().lower()
    new = TYPE_MAP.get(key, key if key in VALID else "string")
    if new != t:
        fixes["count"] += 1
    return new


def walk(node: Any) -> None:
    """Recursively normalize any 'type' string values in a JSON-Schema subtree."""
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == "type":
                if isinstance(v, str):
                    node[k] = norm_type(v)
                elif isinstance(v, list):
                    node[k] = [norm_type(x) for x in v]
                else:
                    walk(v)
            else:
                walk(v)
    elif isinstance(node, list):
        for x in node:
            walk(x)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/eval_pool_114items.jsonl")
    if not path.exists():
        raise FileNotFoundError(path)

    raw_backup = path.with_suffix(".raw.jsonl")
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if not raw_backup.exists():
        raw_backup.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
        print(f"Backed up original -> {raw_backup}")
    else:
        print(f"Backup already exists -> {raw_backup} (not overwriting)")

    items_touched = 0
    for rec in records:
        before = fixes["count"]
        for tool in rec.get("tools", []):
            params = tool.get("function", {}).get("parameters")
            if params is not None:
                walk(params)
        if fixes["count"] > before:
            items_touched += 1

    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )

    print(f"Sanitized {len(records)} items: normalized {fixes['count']} type values "
          f"across {items_touched} items.")
    print(f"Rewrote -> {path}")


if __name__ == "__main__":
    main()
