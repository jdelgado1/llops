#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llmops.fabric_integration import FabricExporter

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
SRC = Path("artifacts/tool-sft-20260621-140128-safe.jsonl")
VERSION = "dev3-seed-v1-sft"
BASE = f"{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports/{VERSION}"

records = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
exporter = FabricExporter(onelake_workspace=WORKSPACE, onelake_lakehouse=LAKEHOUSE)
content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dataset_version": VERSION,
    "example_count": len(records),
    "status": "seed_sft_for_gpt_4_1_nano",
    "base_model": "gpt-4.1-nano",
    "training_type": "Global",
    "train_path": f"Files/llmops/foundry_exports/{VERSION}/train.jsonl",
}
exporter._write_file(Path(f"onelake://{WORKSPACE}/{BASE}/train.jsonl"), content)
exporter._write_file(
    Path(f"onelake://{WORKSPACE}/{BASE}/manifest.json"),
    json.dumps(manifest, indent=2).encode("utf-8"),
)
print(f"Uploaded {len(records)} seed SFT rows to Files/llmops/foundry_exports/{VERSION}/train.jsonl")
