#!/usr/bin/env python
"""
Build the cumulative SFT dataset for "building on" the initial training:
    original base SFT  +  retrain-v2 corrections (SFT-converted)

Uploads to OneLake so it's selectable in the Foundry fine-tune UI (catalog):
    Files/llmops/foundry_exports/retrain-v2-cumulative/train.jsonl

Use this with Option B (recommended): base model = Qwen3-32B, train on the
cumulative set -> the student keeps prior skills AND fixes the 19 failures.
"""
import json
import logging
from pathlib import Path

from llmops.fabric_integration import FabricExporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_SFT = Path("artifacts/tool-sft-20260621-140128-safe.jsonl")
CORRECTIONS_SFT = Path("artifacts/retrain-v2-sft/train.jsonl")  # produced by convert_retrain_v2_to_sft.py
OUT_LOCAL = Path("artifacts/retrain-v2-cumulative/train.jsonl")

ONELAKE_WORKSPACE = "Fine Tune Demo"
ONELAKE_LAKEHOUSE = "lh_llmops"
ONELAKE_REL = "Files/llmops/foundry_exports/retrain-v2-cumulative/train.jsonl"


def _read(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    base = _read(BASE_SFT)
    corrections = _read(CORRECTIONS_SFT)
    cumulative = base + corrections
    logger.info(f"base={len(base)} + corrections={len(corrections)} = {len(cumulative)} cumulative")

    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LOCAL, "w", encoding="utf-8") as f:
        for r in cumulative:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"✅ Wrote local cumulative SFT: {OUT_LOCAL}")

    exporter = FabricExporter(
        onelake_workspace=ONELAKE_WORKSPACE,
        onelake_lakehouse=ONELAKE_LAKEHOUSE,
    )
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cumulative).encode("utf-8")
    fabric_path = Path(f"onelake://{ONELAKE_WORKSPACE}/{ONELAKE_LAKEHOUSE}.Lakehouse/{ONELAKE_REL}")
    exporter._write_file(fabric_path, content)

    logger.info(f"\n✅ Uploaded to OneLake (selectable in Foundry catalog):")
    logger.info(f"   {ONELAKE_REL}")
    logger.info(f"\nFoundry UI: base model = Qwen3-32B, training data = this file, ~3 epochs.")


if __name__ == "__main__":
    main()
