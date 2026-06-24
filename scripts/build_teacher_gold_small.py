#!/usr/bin/env python
"""Build a small teacher-gold SFT train/eval set from hard BFCL items.

This is the compact distillation demo dataset:
  - gpt-5.4 generates the assistant tool calls (teacher gold)
  - first N good examples -> train.jsonl
  - next M good examples -> eval.jsonl (held out)
  - upload both to OneLake foundry_exports/dev3-teacher-gold-small/
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
from llmops.models import invoke_model
from llmops.tool_models import TOOL_SYSTEM

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
VERSION = "dev3-teacher-gold-small"
POOL = ROOT / "artifacts" / "eval_pool_hard.jsonl"
OUT_DIR = ROOT / "artifacts" / VERSION
TRAIN_N = 20
EVAL_N = 10
TEACHER = "gpt-5.4"


def to_sft_record(item: dict[str, Any], calls: list[dict[str, Any]], source_index: int) -> dict[str, Any]:
    # Keep only non-assistant prompt turns, and ensure a stable system instruction.
    messages = [m for m in item.get("messages", []) if m.get("role") != "assistant"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": TOOL_SYSTEM}, *messages]

    assistant_calls = []
    for i, call in enumerate(calls):
        assistant_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            },
        })

    return {
        "messages": [*messages, {"role": "assistant", "tool_calls": assistant_calls}],
        "tools": item.get("tools", []),
        "metadata": {
            "dataset_version": VERSION,
            "source_pool": str(POOL.name),
            "source_index": source_index,
            "category": item.get("category"),
            "teacher_model": TEACHER,
            "gold_source": "teacher_tool_calls",
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def upload(path: Path, records: list[dict[str, Any]], exporter: FabricExporter) -> None:
    rel = f"{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports/{VERSION}/{path.name}"
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    exporter._write_file(Path(f"onelake://{WORKSPACE}/{rel}"), content)
    print(f"Uploaded -> Files/llmops/foundry_exports/{VERSION}/{path.name}")


def main() -> None:
    items = [json.loads(l) for l in POOL.read_text(encoding="utf-8").splitlines() if l.strip()]
    needed = TRAIN_N + EVAL_N
    records: list[dict[str, Any]] = []

    print(f"Generating {needed} teacher-gold examples from {POOL} using {TEACHER}...")
    for idx, item in enumerate(items):
        if len(records) >= needed:
            break
        try:
            result = invoke_model(TEACHER, item.get("messages", []), item.get("tools", []))
            calls = result.get("tool_calls", [])
            if not calls:
                print(f"  skip {idx}: teacher returned no tool calls")
                continue
            records.append(to_sft_record(item, calls, idx))
            if len(records) % 10 == 0:
                print(f"  collected {len(records)}/{needed}")
        except Exception as e:
            print(f"  skip {idx}: {str(e)[:120]}")

    if len(records) < needed:
        raise RuntimeError(f"Only collected {len(records)} teacher-gold examples; needed {needed}")

    train = records[:TRAIN_N]
    eval_records = records[TRAIN_N:needed]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "eval.jsonl", eval_records)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": VERSION,
        "teacher_model": TEACHER,
        "base_model": "gpt-4.1-nano",
        "training_type": "Global",
        "train_count": len(train),
        "eval_count": len(eval_records),
        "source_pool": str(POOL.name),
        "train_path": f"Files/llmops/foundry_exports/{VERSION}/train.jsonl",
        "eval_path": f"Files/llmops/foundry_exports/{VERSION}/eval.jsonl",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    exporter = FabricExporter(onelake_workspace=WORKSPACE, onelake_lakehouse=LAKEHOUSE)
    upload(OUT_DIR / "train.jsonl", train, exporter)
    upload(OUT_DIR / "eval.jsonl", eval_records, exporter)
    exporter._write_file(
        Path(f"onelake://{WORKSPACE}/{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports/{VERSION}/manifest.json"),
        json.dumps(manifest, indent=2).encode("utf-8"),
    )
    print(f"Wrote local -> {OUT_DIR}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
