#!/usr/bin/env python
"""Run one dev3 text-SFT retraining loop from Blu's OneLake export.

This is the clean repeatable entrypoint for the working demo path.

Example:
  python scripts/run_dev3_retrain_once.py \
    --dataset-version dev3-student-v2 \
    --base-model <previous fine-tuned model id> \
    --deployment-name gpt-41-nano-student-v2

What it does:
  1. Downloads Files/llmops/foundry_exports/<dataset-version>/train.jsonl
  2. Converts Blu's rows to schema-free <tool_call> text SFT
  3. Uploads the JSONL through the Azure OpenAI account endpoint
  4. Submits GlobalStandard fine-tuning
  5. Waits for completion
  6. Deploys the fine-tuned model
  7. Writes local state artifact for auditability
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from llmops.dev3_pipeline import (
    convert_blu_rows_to_text_sft,
    deploy_finetuned_model,
    download_onelake_jsonl,
    submit_global_finetune,
    upload_aoai_file,
    wait_for_finetune,
)

DEFAULT_BASE_MODEL = "gpt-4.1-nano-2025-04-14.ft-56718845ff1c4fd1a68a50e7c7800f6d-dev3-text-account-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dev3 retrain loop from Blu's OneLake export")
    parser.add_argument("--dataset-version", required=True, help="Folder under Files/llmops/foundry_exports")
    parser.add_argument("--base-model", default=os.environ.get("CURRENT_STUDENT_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--deployment-name", required=True, help="Deployment name for the new fine-tuned model")
    parser.add_argument("--suffix", default=None, help="Fine-tune suffix; defaults to dataset version")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--workspace", default=os.environ.get("ONELAKE_WORKSPACE", "Fine Tune Demo"))
    parser.add_argument("--lakehouse", default=os.environ.get("ONELAKE_LAKEHOUSE", "lh_llmops"))
    parser.add_argument("--aoai-account", default=os.environ.get("FOUNDRY_AOAI_ACCOUNT", "TuneSEfoundry"))
    parser.add_argument("--resource-group", default=os.environ.get("FOUNDRY_RESOURCE_GROUP", "demo-rg"))
    parser.add_argument("--no-watch", action="store_true", help="Submit job and exit without waiting/deploying")
    args = parser.parse_args()

    suffix = args.suffix or args.dataset_version
    work_dir = Path("artifacts") / args.dataset_version
    source_path = work_dir / "train.jsonl"
    text_path = work_dir / "train_text.jsonl"
    remote_train = f"Files/llmops/foundry_exports/{args.dataset_version}/train.jsonl"

    print(f"Downloading {remote_train}", flush=True)
    rows = download_onelake_jsonl(args.workspace, args.lakehouse, remote_train, source_path)
    print(f"Downloaded {len(rows)} rows -> {source_path}", flush=True)

    convert_blu_rows_to_text_sft(rows, text_path)
    print(f"Converted to text SFT -> {text_path}", flush=True)

    file_id = upload_aoai_file(args.aoai_account, text_path)
    print(f"Uploaded fine-tune file: {file_id}", flush=True)

    job = submit_global_finetune(
        account=args.aoai_account,
        base_model=args.base_model,
        training_file_id=file_id,
        suffix=suffix,
        epochs=args.epochs,
    )
    (work_dir / "fine_tune_job_submitted.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    print(f"Submitted fine-tune job: {job['id']} status={job['status']}", flush=True)

    if args.no_watch:
        return

    job = wait_for_finetune(args.aoai_account, job["id"])
    (work_dir / "fine_tune_job_final.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    if job.get("status") != "succeeded":
        raise RuntimeError(f"Fine-tune did not succeed: {job}")

    fine_tuned_model = job["fine_tuned_model"]
    print(f"Deploying {fine_tuned_model} as {args.deployment_name}", flush=True)
    deploy_finetuned_model(args.aoai_account, args.resource_group, args.deployment_name, fine_tuned_model)

    state = {
        "dataset_version": args.dataset_version,
        "base_model": args.base_model,
        "fine_tuned_model": fine_tuned_model,
        "deployment_name": args.deployment_name,
        "job_id": job["id"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (work_dir / "pipeline_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
