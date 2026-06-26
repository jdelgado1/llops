#!/usr/bin/env python
"""
Convert Blu's retrain-v2 dataset from eval-reference format to Foundry SFT format.

Blu's format (NOT trainable):
  {"messages": [{"role":"user","content":"..."}],
   "tool_calls": [{"name":"f","arguments":{"p":["v"], "port":[5432]}}],
   "tools":[...]}

Foundry SFT format (trainable):
  {"messages": [
     {"role":"system","content":"You are a TMG..."},
     {"role":"user","content":"..."},
     {"role":"assistant","tool_calls":[
        {"id":"call_0","type":"function",
         "function":{"name":"f","arguments":"{\"p\": \"v\", \"port\": 5432}"}}]}],
   "tools":[...]}

Transformations:
  1. Prepend canonical system message.
  2. Move top-level tool_calls into an assistant message.
  3. Convert arguments dict-of-lists (AST acceptable values) -> scalar values
     by taking the first non-empty acceptable value (preserving type).
  4. Serialize arguments as a JSON string (Foundry SFT requirement).
  5. Drop metadata (match known-good SFT schema exactly).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from llmops.fabric_integration import FabricExporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a TMG (Telco/Media/Gaming) operations & support agent. "
    "Use the provided tools to fulfill the user's request. "
    "Call the correct function(s) with correct arguments. "
    "Do not ask follow-up questions if the request is answerable with the tools."
)

SRC = Path("artifacts/train.jsonl")
OUT = Path("artifacts/retrain-v2-sft/train.jsonl")

ONELAKE_WORKSPACE = "Fine Tune Demo"
ONELAKE_LAKEHOUSE = "lh_llmops"
ONELAKE_DIR = "lh_llmops.Lakehouse/Files/llmops/foundry_exports/retrain-v2-sft"


def _scalarize(value: Any) -> Any:
    """Convert an AST acceptable-values list into a single concrete value."""
    if isinstance(value, list):
        if not value:
            return None
        # Prefer the first non-empty value (handles cases like ["all", ""])
        for v in value:
            if v != "" and v is not None:
                return v
        return value[0]
    return value


def _scalarize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _scalarize(v) for k, v in args.items()}


def convert_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # User content (Blu's messages list holds only the user turn)
    user_content = ""
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            user_content = m.get("content", "")
            break

    # Build assistant tool_calls in Foundry SFT shape
    assistant_tool_calls: List[Dict[str, Any]] = []
    for i, tc in enumerate(rec.get("tool_calls", [])):
        scalar_args = _scalarize_args(tc.get("arguments", {}))
        assistant_tool_calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(scalar_args, ensure_ascii=False),
                },
            }
        )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "tool_calls": assistant_tool_calls},
        ],
        "tools": rec.get("tools", []),
    }


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source not found: {SRC}")

    src_records = []
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                src_records.append(json.loads(line))

    logger.info(f"Loaded {len(src_records)} source records from {SRC}")

    converted = [convert_record(r) for r in src_records]

    # Write locally
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"✅ Wrote {len(converted)} SFT records to {OUT}")

    # Show first two converted examples for verification
    logger.info("\n--- Sample converted records ---")
    for rec in converted[:2]:
        tc = rec["messages"][2]["tool_calls"][0]["function"]
        logger.info(f"  user: {rec['messages'][1]['content'][:70]}...")
        logger.info(f"  -> {tc['name']}({tc['arguments']})")

    # Upload to OneLake
    logger.info(f"\n🔄 Uploading SFT-ready file to OneLake: {ONELAKE_DIR}/train.jsonl")
    exporter = FabricExporter(
        onelake_workspace=ONELAKE_WORKSPACE,
        onelake_lakehouse=ONELAKE_LAKEHOUSE,
    )
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted)
    fabric_path = Path(f"onelake://{ONELAKE_WORKSPACE}/{ONELAKE_DIR}/train.jsonl")
    exporter._write_file(fabric_path, content.encode("utf-8"))

    logger.info(f"\n✅ Done. Foundry-ready dataset at:")
    logger.info(f"   OneLake: Files/llmops/foundry_exports/retrain-v2-sft/train.jsonl")
    logger.info(f"   Local:   {OUT}")


if __name__ == "__main__":
    main()
