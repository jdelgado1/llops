#!/usr/bin/env python
"""
Autonomous retraining-loop orchestrator.

Chains the whole distillation loop so it can run unattended (Task Scheduler /
cron / Fabric pipeline):

  1. DETECT  new retrain dataset in OneLake foundry_exports/retrain-*/ (vs state)
  2. PULL    train.jsonl + manifest.json from OneLake
  3. CONVERT eval-reference format -> Foundry SFT format
  4. MERGE   (optional) append to a base SFT file for stability
  5. FINETUNE submit a managed Foundry SFT job and watch to completion
  6. DEPLOY  (optional, gated) serverless-deploy the fine-tuned model via `az`
  7. EVAL    run the 3-way eval and export results to Fabric
  8. GATE    compare new student AST vs previous; promote or hold; persist state

Designed to be idempotent: if the newest dataset_version has already been
processed, the loop is a no-op.

AUTH for unattended runs (no interactive login):
  Set a service principal in the environment (DefaultAzureCredential picks these
  up automatically):
    AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
  The SP must have access to BOTH the Foundry project and the Fabric workspace.

USAGE
  # one pass (typical for a scheduled task)
  python scripts/run_retrain_loop.py --once

  # continuous loop, check every 6h
  python scripts/run_retrain_loop.py --loop --interval-hours 6

  # see what it would do without changing anything
  python scripts/run_retrain_loop.py --once --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- make src/ and scripts/ importable -------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

from llmops.finetune import submit_finetune
from llmops.config import get_settings
# reuse the conversion logic we already wrote
from convert_retrain_v2_to_sft import convert_record, SYSTEM_PROMPT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("retrain-loop")

# --- configuration (override via env / flags) ------------------------------
ONELAKE_ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
ONELAKE_WORKSPACE = "Fine Tune Demo"
ONELAKE_LAKEHOUSE = "lh_llmops"
EXPORTS_DIR = f"{ONELAKE_LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports"

STATE_FILE = ROOT / "artifacts" / ".retrain_loop_state.json"
WORK_DIR = ROOT / "artifacts" / "retrain_loop"
BASE_SFT_FILE = ROOT / "artifacts" / "tool-sft-20260621-140128-safe.jsonl"

DEFAULT_EPOCHS = 3
EVAL_POOL = ROOT / "artifacts" / "eval_pool_114items.jsonl"


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------
def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "processed_versions": [],
        "current_student_model": None,  # latest fine-tuned model id (continuation base)
        "last_student_ast": None,
        "history": [],
    }


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# OneLake helpers
# ---------------------------------------------------------------------------
def _fs_client():
    cred = DefaultAzureCredential()
    svc = DataLakeServiceClient(account_url=ONELAKE_ACCOUNT_URL, credential=cred)
    return svc.get_file_system_client(ONELAKE_WORKSPACE)


def find_latest_retrain(fs) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (retrain_dir_name, manifest) for the newest retrain-* with a manifest."""
    candidates: List[Tuple[str, str]] = []  # (created_at, dir_name)
    seen_dirs = set()
    for path in fs.get_paths(path=EXPORTS_DIR, recursive=True):
        if path.is_directory:
            continue
        rel = path.name.replace(f"{EXPORTS_DIR}/", "")
        parts = rel.split("/")
        if len(parts) != 2 or parts[1] != "manifest.json":
            continue
        dir_name = parts[0]
        if dir_name in seen_dirs:
            continue
        seen_dirs.add(dir_name)
        manifest_bytes = fs.get_file_client(path.name).download_file().readall()
        manifest = json.loads(manifest_bytes)
        created = manifest.get("created_at", "")
        candidates.append((created, dir_name))

    if not candidates:
        return None
    candidates.sort()  # ISO timestamps sort lexically
    _, latest_dir = candidates[-1]
    manifest = json.loads(
        fs.get_file_client(f"{EXPORTS_DIR}/{latest_dir}/manifest.json")
        .download_file()
        .readall()
    )
    return latest_dir, manifest


def download_train_jsonl(fs, retrain_dir: str, dest: Path) -> List[Dict[str, Any]]:
    data = (
        fs.get_file_client(f"{EXPORTS_DIR}/{retrain_dir}/train.jsonl")
        .download_file()
        .readall()
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    records = [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]
    logger.info(f"Downloaded {len(records)} records -> {dest}")
    return records


def upload_jsonl(fs, remote_rel: str, records: List[Dict[str, Any]]) -> str:
    """Upload JSONL under the lakehouse; remote_rel is relative to the lakehouse Files root."""
    full = f"{ONELAKE_LAKEHOUSE}.Lakehouse/{remote_rel}"
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    dir_path = "/".join(full.split("/")[:-1])
    fs.get_directory_client(dir_path).create_directory()
    fs.get_file_client(full).upload_data(content, overwrite=True)
    logger.info(f"Uploaded {len(records)} records -> OneLake {full}")
    return full


# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------
def stage_convert(src_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = [convert_record(r) for r in src_records]
    logger.info(f"Converted {len(converted)} records to Foundry SFT format")
    return converted


def stage_merge(sft_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append the new corrections to the base SFT set for stability (avoid forgetting)."""
    if not BASE_SFT_FILE.exists():
        logger.warning(f"Base SFT file not found ({BASE_SFT_FILE}); training on corrections only")
        return sft_records
    base = [
        json.loads(l)
        for l in BASE_SFT_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    merged = base + sft_records
    logger.info(f"Merged base({len(base)}) + corrections({len(sft_records)}) = {len(merged)}")
    return merged


def stage_finetune(
    training_file: Path,
    base_model: str,
    suffix: str,
    epochs: int,
    training_type: Optional[str],
    dry_run: bool,
) -> Optional[str]:
    if dry_run:
        logger.info(f"[dry-run] would submit fine-tune: file={training_file} "
                    f"base={base_model} suffix={suffix} epochs={epochs} "
                    f"training_type={training_type or 'default'}")
        return None
    job = submit_finetune(
        training_file=str(training_file),
        model=base_model,
        suffix=suffix,
        epochs=epochs,
        training_type=training_type,
        watch=True,
    )
    fine_tuned = getattr(job, "fine_tuned_model", None)
    if getattr(job, "status", None) != "succeeded" or not fine_tuned:
        logger.error(f"Fine-tune did not succeed: status={getattr(job,'status',None)}")
        return None
    logger.info(f"Fine-tune succeeded: {fine_tuned}")
    return fine_tuned


def stage_deploy(fine_tuned_model: str, deployment_name: str, dry_run: bool) -> Optional[str]:
    """
    Serverless-deploy the fine-tuned model via `az` IF the AOAI account + RG are
    configured in the environment. Returns the deployment name on success.

    Requires env:
      FOUNDRY_AOAI_ACCOUNT  - Azure OpenAI / AI Services account name
      FOUNDRY_RESOURCE_GROUP- resource group of that account
    """
    import os
    account = os.environ.get("FOUNDRY_AOAI_ACCOUNT")
    rg = os.environ.get("FOUNDRY_RESOURCE_GROUP")
    if not account or not rg:
        logger.warning(
            "Deploy skipped: set FOUNDRY_AOAI_ACCOUNT and FOUNDRY_RESOURCE_GROUP to "
            "auto-deploy. fine_tuned_model=%s", fine_tuned_model
        )
        return None

    cmd = [
        shutil.which("az") or shutil.which("az.cmd") or "az",
        "cognitiveservices", "account", "deployment", "create",
        "--name", account,
        "--resource-group", rg,
        "--deployment-name", deployment_name,
        "--model-name", fine_tuned_model,
        "--model-version", "1",
        "--model-format", "OpenAI",
        "--sku-name", "GlobalStandard",
        "--sku-capacity", "50",
    ]
    if dry_run:
        logger.info("[dry-run] would deploy: %s", " ".join(cmd))
        return deployment_name
    logger.info("Deploying fine-tuned model: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Deploy failed: %s", res.stderr.strip())
        return None
    logger.info("Deployed as %s", deployment_name)
    return deployment_name


def stage_eval(
    student_deployment: str,
    baseline_deployment: str,
    eval_run_name: str,
    dry_run: bool,
) -> bool:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_real_integration.py"),
        "--eval-pool-jsonl", str(EVAL_POOL),
        "--baseline-deployment", baseline_deployment,
        "--student-deployment", student_deployment,
        "--eval-run-name", eval_run_name,
        "--onelake-workspace", ONELAKE_WORKSPACE,
    ]
    if dry_run:
        logger.info("[dry-run] would eval: %s", " ".join(cmd))
        return True
    logger.info("Running eval: %s", " ".join(cmd))
    env = {"PYTHONPATH": str(ROOT / "src")}
    import os
    full_env = {**os.environ, **env}
    res = subprocess.run(cmd, env=full_env)
    return res.returncode == 0


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def run_once(epochs: int, dry_run: bool, merge: bool) -> bool:
    """Returns True if a new dataset was processed, False if no-op.

    CONTINUOUS fine-tuning (dev3): each iteration's base model is the PREVIOUS
    fine-tuned student (state['current_student_model']); the very first run uses
    settings.base_finetune_model (gpt-4.1-nano). Training data is the new delta
    (corrections) only — the model already carries prior knowledge in its weights.
    """
    settings = get_settings()
    state = load_state()
    fs = _fs_client()

    latest = find_latest_retrain(fs)
    if latest is None:
        logger.info("No retrain datasets found in OneLake. Nothing to do.")
        return False
    retrain_dir, manifest = latest
    version = manifest.get("dataset_version", retrain_dir)

    if version in state["processed_versions"]:
        logger.info(f"Latest dataset '{version}' already processed. No-op.")
        return False

    # CONTINUATION: base = previous fine-tuned student, else the seed base model
    prev_student = state.get("current_student_model")
    base_model = prev_student or settings.base_finetune_model
    logger.info(f"=== Processing new retrain dataset: {version} ({retrain_dir}) ===")
    logger.info(f"Continuation base model: {base_model} "
                f"({'previous student' if prev_student else 'seed base'})")
    logger.info(f"manifest: {json.dumps(manifest)}")

    # 2. PULL
    src_path = WORK_DIR / version / "source_train.jsonl"
    src_records = download_train_jsonl(fs, retrain_dir, src_path)

    # 3. CONVERT
    sft_records = stage_convert(src_records)

    # 4. MERGE (optional). For continuation the default is delta-only training;
    #    --merge re-includes the base SFT (use when training from the seed base).
    final_records = stage_merge(sft_records) if merge else sft_records
    training_file = WORK_DIR / version / "train_sft.jsonl"
    training_file.parent.mkdir(parents=True, exist_ok=True)
    training_file.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final_records),
        encoding="utf-8",
    )
    logger.info(f"Training file ready: {training_file} ({len(final_records)} records)")

    # also publish the SFT-ready file back to OneLake for provenance
    if not dry_run:
        upload_jsonl(
            fs,
            f"Files/llmops/foundry_exports/{version}-sft/train.jsonl",
            final_records,
        )

    # 5. FINETUNE (continue from base_model)
    suffix = f"{version}".replace("_", "-")[:40]
    fine_tuned = stage_finetune(
        training_file, base_model, suffix, epochs,
        settings.finetune_training_type, dry_run,
    )
    if dry_run:
        return True
    if not fine_tuned:
        logger.error("Stopping: fine-tune failed.")
        return True

    # 6. DEPLOY (gated)
    deployment_name = f"{settings.base_finetune_model}-{suffix}".replace(".", "")[:63]
    deployed = stage_deploy(fine_tuned, deployment_name, dry_run)
    student_deployment = deployed or fine_tuned  # serve by FT model id if not separately deployed

    # 7. EVAL — compare the new student against the PREVIOUS student (or seed base)
    baseline_for_eval = (
        settings.baseline_deployment or prev_student or settings.base_finetune_model
    )
    eval_run_name = f"auto-{version}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    eval_ok = stage_eval(student_deployment, baseline_for_eval, eval_run_name, dry_run)

    # 8. GATE / persist state — advance the continuation pointer
    record = {
        "dataset_version": version,
        "base_model": base_model,
        "fine_tuned_model": fine_tuned,
        "deployment": student_deployment,
        "eval_baseline": baseline_for_eval,
        "eval_run_name": eval_run_name,
        "eval_ok": eval_ok,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    state["processed_versions"].append(version)
    state["current_student_model"] = fine_tuned  # next iteration continues from here
    state["history"].append(record)
    save_state(state)
    logger.info(f"=== Done: {json.dumps(record)} ===")
    logger.info("Continuation pointer advanced -> next retrain will build on %s", fine_tuned)
    logger.info("Promotion gate: review eval_results in Fabric; if new student AST "
                ">= previous, set STUDENT_FINETUNED_DEPLOYMENT in .env to %s",
                student_deployment)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Autonomous retraining-loop orchestrator")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run a single pass")
    mode.add_argument("--loop", action="store_true", help="run continuously")
    ap.add_argument("--interval-hours", type=float, default=6.0,
                    help="poll interval for --loop (default 6h)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--merge", action="store_true",
                    help="also include the base SFT set (cumulative). Default: "
                         "delta-only, which is correct for CONTINUOUS fine-tuning "
                         "(the base model already carries prior knowledge).")
    ap.add_argument("--dry-run", action="store_true",
                    help="show actions without submitting/deploying/evaluating")
    args = ap.parse_args()

    merge = args.merge

    if args.once:
        run_once(args.epochs, args.dry_run, merge)
        return

    logger.info(f"Starting continuous loop (interval={args.interval_hours}h). Ctrl-C to stop.")
    while True:
        try:
            run_once(args.epochs, args.dry_run, merge)
        except Exception as e:  # keep the daemon alive across transient failures
            logger.exception("Loop iteration failed: %s", e)
        time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    main()
