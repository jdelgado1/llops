#!/usr/bin/env python
"""
Upload the corrected eval_details file to OneLake so Blu can ingest it.

This is the INPUT file for Blu's retraining workflow:
  - Tells Blu which 19 items the student got wrong (student=false)
  - Blu filters these, builds corrected SFT examples
  - Blu writes the OUTPUT to foundry_exports/ (which Foundry's data asset points at)

Target path in OneLake:
  Files/llmops/raw/foundry_evals/baseline-vs-student-corrected/eval_details_*.jsonl
"""

import json
import logging
from pathlib import Path

from llmops.fabric_integration import FabricExporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LOCAL_FILE = Path("artifacts/eval_details_corrected_20260623.jsonl")
EVAL_RUN_NAME = "baseline-vs-student-corrected"
ONELAKE_WORKSPACE = "Fine Tune Demo"
ONELAKE_LAKEHOUSE = "lh_llmops"


def main() -> None:
    logger.info("=" * 60)
    logger.info("Uploading corrected eval_details to OneLake catalog")
    logger.info("=" * 60)

    # Load the local corrected file
    if not LOCAL_FILE.exists():
        raise FileNotFoundError(f"Local file not found: {LOCAL_FILE}")

    records = []
    with open(LOCAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"✅ Loaded {len(records)} records from {LOCAL_FILE}")

    # Verify distribution before upload
    student_true = sum(1 for r in records if r["ast_match_by_model"]["student"])
    student_false = len(records) - student_true
    baseline_true = sum(1 for r in records if r["ast_match_by_model"]["baseline"])
    teacher_true = sum(1 for r in records if r["ast_match_by_model"]["teacher"])

    logger.info(f"\n📊 Distribution check:")
    logger.info(f"   baseline: {baseline_true}/{len(records)} = {baseline_true/len(records)*100:.2f}%")
    logger.info(f"   student:  {student_true}/{len(records)} = {student_true/len(records)*100:.2f}%")
    logger.info(f"   teacher:  {teacher_true}/{len(records)} = {teacher_true/len(records)*100:.2f}%")
    logger.info(f"\n   ⚠️  Student failed items: {student_false} (these are the retrain candidates)")

    failed_ids = [r["eval_item_id"] for r in records if not r["ast_match_by_model"]["student"]]
    logger.info(f"   Failed eval_item_ids: {failed_ids}")

    # Upload to OneLake
    logger.info(f"\n🔄 Uploading to OneLake...")
    logger.info(f"   Workspace: {ONELAKE_WORKSPACE}")
    logger.info(f"   Lakehouse: {ONELAKE_LAKEHOUSE}")
    logger.info(f"   Run name:  {EVAL_RUN_NAME}")

    exporter = FabricExporter(
        onelake_workspace=ONELAKE_WORKSPACE,
        onelake_lakehouse=ONELAKE_LAKEHOUSE,
    )

    details_path = exporter.export_eval_details_jsonl(records, EVAL_RUN_NAME)

    logger.info(f"\n✅ Upload complete!")
    logger.info(f"   OneLake path: {details_path}")
    logger.info(f"\n🎯 Blu can now ingest from:")
    logger.info(f"   Files/llmops/raw/foundry_evals/{EVAL_RUN_NAME}/")
    logger.info(f"\n   Blu's retrain filter (selects 19 failed student items):")
    logger.info(f"     WHERE ast_match_by_model.student = false")
    logger.info(f"\n   Blu then writes corrected SFT to:")
    logger.info(f"     Files/llmops/foundry_exports/golden-drift-corrected-<date>.jsonl")
    logger.info(f"   ...which Foundry's 'foundry-retraining-candidates' data asset auto-detects.")


if __name__ == "__main__":
    main()
