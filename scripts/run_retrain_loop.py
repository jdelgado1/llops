#!/usr/bin/env python
"""ACA-friendly retraining loop entrypoint.

This file is now a thin wrapper around the proven dev3 pipeline path:
- Detect newest OneLake `Files/llmops/foundry_exports/<version>/manifest.json`.
- Download `<version>/train.jsonl`.
- Convert Blu rows to schema-free `<tool_call>{...}</tool_call>` text SFT.
- Upload through the Azure OpenAI account endpoint.
- Submit `trainingType=GlobalStandard` fine-tune from the current student model.
- Optionally wait and deploy the resulting fine-tuned model.

The previous native tool-schema fine-tune path is intentionally not used because
Azure OpenAI preprocessing rejected native tool schemas for gpt-4.1-nano.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

from llmops.dev3_pipeline import (
    ONELAKE_ACCOUNT_URL,
    convert_blu_rows_to_text_sft,
    deploy_finetuned_model,
    download_onelake_jsonl,
    submit_global_finetune,
    upload_aoai_file,
    wait_for_finetune,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("retrain-loop")

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "artifacts" / ".retrain_loop_state.json"
WORK_ROOT = ROOT / "artifacts" / "retrain_loop"

WORKSPACE = os.environ.get("ONELAKE_WORKSPACE", "Fine Tune Demo")
LAKEHOUSE = os.environ.get("ONELAKE_LAKEHOUSE", "lh_llmops")
EXPORTS_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports"
AOAI_ACCOUNT = os.environ.get("FOUNDRY_AOAI_ACCOUNT", "TuneSEfoundry")
RESOURCE_GROUP = os.environ.get("FOUNDRY_RESOURCE_GROUP", "demo-rg")
SEED_BASE_MODEL = os.environ.get("BASE_FINETUNE_MODEL", "gpt-4.1-nano")
DEFAULT_CURRENT_STUDENT = os.environ.get("CURRENT_STUDENT_MODEL") or os.environ.get("CURRENT_STUDENT_FINE_TUNED_MODEL")


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"processed_versions": [], "current_student_model": DEFAULT_CURRENT_STUDENT, "history": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def fs_client():
    credential = DefaultAzureCredential(
        managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID") or None,
        exclude_interactive_browser_credential=True,
    )
    return DataLakeServiceClient(account_url=ONELAKE_ACCOUNT_URL, credential=credential).get_file_system_client(WORKSPACE)


def find_latest_export() -> tuple[str, dict[str, Any]] | None:
    fs = fs_client()
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for path in fs.get_paths(path=EXPORTS_DIR, recursive=True):
        if path.is_directory or not path.name.endswith("/manifest.json"):
            continue
        rel = path.name.replace(f"{EXPORTS_DIR}/", "")
        parts = rel.split("/")
        if len(parts) != 2:
            continue
        version = parts[0]
        manifest = json.loads(fs.get_file_client(path.name).download_file().readall())
        created = manifest.get("created_at") or manifest.get("created") or ""
        candidates.append((created, version, manifest))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, version, manifest = candidates[-1]
    return version, manifest


def deployment_name_for(version: str) -> str:
    override = os.environ.get("NEXT_STUDENT_DEPLOYMENT")
    if override:
        return override
    safe = version.replace("_", "-").replace(".", "-")
    return f"gpt-41-nano-{safe}"[:63]


def run_once(args: argparse.Namespace) -> bool:
    latest = find_latest_export()
    if latest is None:
        logger.info("No foundry_exports manifest found. Nothing to do.")
        return False

    version, manifest = latest
    state = load_state()
    if version in state.get("processed_versions", []):
        logger.info("Latest dataset %s already processed. No-op.", version)
        return False

    base_model = args.base_model or state.get("current_student_model") or SEED_BASE_MODEL
    deployment_name = args.deployment_name or deployment_name_for(version)
    work_dir = WORK_ROOT / version
    source_path = work_dir / "train.jsonl"
    text_path = work_dir / "train_text.jsonl"
    remote_train = f"Files/llmops/foundry_exports/{version}/train.jsonl"

    logger.info("Processing dataset version=%s", version)
    logger.info("Manifest=%s", json.dumps(manifest))
    logger.info("Base model=%s", base_model)
    logger.info("Deployment name=%s", deployment_name)

    if args.dry_run:
        logger.info("[dry-run] would download %s", remote_train)
        logger.info("[dry-run] would convert to text SFT and submit GlobalStandard fine-tune")
        return True

    rows = download_onelake_jsonl(WORKSPACE, LAKEHOUSE, remote_train, source_path)
    logger.info("Downloaded %d rows -> %s", len(rows), source_path)
    convert_blu_rows_to_text_sft(rows, text_path)
    logger.info("Converted -> %s", text_path)

    file_id = upload_aoai_file(AOAI_ACCOUNT, text_path)
    logger.info("Uploaded fine-tune file id=%s", file_id)

    job = submit_global_finetune(
        account=AOAI_ACCOUNT,
        base_model=base_model,
        training_file_id=file_id,
        suffix=version,
        epochs=args.epochs,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "fine_tune_job_submitted.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    logger.info("Submitted job %s status=%s", job["id"], job["status"])

    if args.no_watch:
        return True

    job = wait_for_finetune(AOAI_ACCOUNT, job["id"], poll_seconds=args.poll_seconds)
    (work_dir / "fine_tune_job_final.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    if job.get("status") != "succeeded":
        raise RuntimeError(f"Fine-tune failed: {job}")

    fine_tuned_model = job["fine_tuned_model"]
    deploy_finetuned_model(AOAI_ACCOUNT, RESOURCE_GROUP, deployment_name, fine_tuned_model, capacity=args.capacity)

    record = {
        "dataset_version": version,
        "base_model": base_model,
        "fine_tuned_model": fine_tuned_model,
        "deployment": deployment_name,
        "job_id": job["id"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    state.setdefault("processed_versions", []).append(version)
    state["current_student_model"] = fine_tuned_model
    state["current_student_deployment"] = deployment_name
    state.setdefault("history", []).append(record)
    save_state(state)
    logger.info("Done: %s", json.dumps(record))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dev3 retraining loop once or continuously")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-hours", type=float, default=6.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--deployment-name", default=None)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.once:
        run_once(args)
        return

    while True:
        try:
            run_once(args)
        except Exception:
            logger.exception("Loop iteration failed")
        time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    main()
