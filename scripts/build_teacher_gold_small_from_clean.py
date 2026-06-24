#!/usr/bin/env python
"""Build a small teacher-anchored SFT train/eval set from the clean pool.

The clean pool contains only items where gpt-5.4 matched the reference. Therefore
reference_tool_calls are valid teacher-gold labels without another model call.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.fabric_integration import FabricExporter
from llmops.tool_models import TOOL_SYSTEM

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
VERSION = "dev3-teacher-gold-small"
POOL = ROOT / "artifacts" / "eval_pool_clean_teacher.jsonl"
OUT_DIR = ROOT / "artifacts" / VERSION
TRAIN_N = 20
EVAL_N = 10


def scalarize(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            if item not in (None, ""):
                return item
        return value[0] if value else None
    return value


def sft_record(item: dict[str, Any], source_index: int) -> dict[str, Any]:
    messages = [m for m in item.get("messages", []) if m.get("role") != "assistant"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": TOOL_SYSTEM}, *messages]

    assistant_calls = []
    for i, call in enumerate(item.get("reference_tool_calls", [])):
        args = {k: scalarize(v) for k, v in call.get("arguments", {}).items()}
        assistant_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    return {
        "messages": [*messages, {"role": "assistant", "tool_calls": assistant_calls}],
        "tools": item.get("tools", []),
        "metadata": {
            "dataset_version": VERSION,
            "source_pool": POOL.name,
            "source_index": source_index,
            "gold_source": "teacher_anchored_reference_call",
            "teacher_model": "gpt-5.4",
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def upload(exporter: FabricExporter, name: str, records: list[dict[str, Any]]) -> None:
    rel = f"{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports/{VERSION}/{name}"
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    exporter._write_file(Path(f"onelake://{WORKSPACE}/{rel}"), content)
    print(f"Uploaded -> Files/llmops/foundry_exports/{VERSION}/{name}")


def main() -> None:
    items = [json.loads(l) for l in POOL.read_text(encoding="utf-8").splitlines() if l.strip()]
    needed = TRAIN_N + EVAL_N
    if len(items) < needed:
        raise RuntimeError(f"Need {needed} clean items; found {len(items)}")

    records = [sft_record(item, i) for i, item in enumerate(items[:needed])]
    train, eval_records = records[:TRAIN_N], records[TRAIN_N:]

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "eval.jsonl", eval_records)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": VERSION,
        "teacher_model": "gpt-5.4",
        "base_model": "gpt-4.1-nano",
        "training_type": "Global",
        "train_count": len(train),
        "eval_count": len(eval_records),
        "source_pool": POOL.name,
        "train_path": f"Files/llmops/foundry_exports/{VERSION}/train.jsonl",
        "eval_path": f"Files/llmops/foundry_exports/{VERSION}/eval.jsonl",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    exporter = FabricExporter(onelake_workspace=WORKSPACE, onelake_lakehouse=LAKEHOUSE)
    upload(exporter, "train.jsonl", train)
    upload(exporter, "eval.jsonl", eval_records)
    exporter._write_file(
        Path(f"onelake://{WORKSPACE}/{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports/{VERSION}/manifest.json"),
        json.dumps(manifest, indent=2).encode("utf-8"),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
