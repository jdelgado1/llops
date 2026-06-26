#!/usr/bin/env python
"""
dev3 bootstrap: seed the CONTINUOUS retraining loop on a small model.

Why this exists
---------------
Qwen3-32B (serverless) can't be used as a base for another fine-tune, so the loop
had to retrain from scratch each time. gpt-4.1-nano is a small Azure OpenAI model
that DOES support continuous fine-tuning, so we can demonstrate the real loop:

    gpt-4.1-nano (base)  --SFT-->  student-v1  --SFT(delta)-->  student-v2  --> ...

This script does the one-time seed (student-v1) and records the continuation
pointer in artifacts/.retrain_loop_state.json. After this, every new retrain
dataset is handled by:  python scripts/run_retrain_loop.py --once

Steps
-----
  1. (optional) deploy the base gpt-4.1-nano chat deployment for a baseline
  2. (optional) baseline eval of the base model -> Fabric
  3. fine-tune student-v1 from gpt-4.1-nano on the initial SFT set
  4. record current_student_model = <student-v1> so the loop continues from it

Usage
-----
  # preview everything (no jobs submitted)
  python scripts/bootstrap_dev3.py --dry-run

  # seed the loop for real (submits a fine-tune; small + cheap on nano)
  python scripts/bootstrap_dev3.py --epochs 2

  # also deploy the base model + run a baseline eval first
  python scripts/bootstrap_dev3.py --epochs 2 --deploy-baseline --baseline-eval
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmops.config import get_settings
from llmops.finetune import submit_finetune

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("bootstrap-dev3")

STATE_FILE = ROOT / "artifacts" / ".retrain_loop_state.json"
INITIAL_SFT = ROOT / "artifacts" / "tool-sft-20260621-140128-safe.jsonl"
EVAL_POOL = ROOT / "artifacts" / "eval_pool_114items.jsonl"

# Foundry resource (blugotlieb tenant). Override via env if needed.
FOUNDRY_ACCOUNT = os.environ.get("FOUNDRY_AOAI_ACCOUNT", "TuneSEfoundry")
FOUNDRY_RG = os.environ.get("FOUNDRY_RESOURCE_GROUP", "demo-rg")
ONELAKE_WORKSPACE = "Fine Tune Demo"


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"processed_versions": [], "current_student_model": None,
            "last_student_ast": None, "history": []}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _az() -> str:
    """Resolve the az CLI (az.cmd on Windows) so subprocess can find it."""
    return shutil.which("az") or shutil.which("az.cmd") or "az"


def deploy_baseline(base_model: str, version: str, dry_run: bool) -> str | None:
    deployment = f"{base_model}-base".replace(".", "")
    cmd = [
        _az(), "cognitiveservices", "account", "deployment", "create",
        "--name", FOUNDRY_ACCOUNT, "--resource-group", FOUNDRY_RG,
        "--deployment-name", deployment,
        "--model-name", base_model, "--model-version", version,
        "--model-format", "OpenAI",
        "--sku-name", "GlobalStandard", "--sku-capacity", "50",
    ]
    if dry_run:
        logger.info("[dry-run] would deploy baseline: %s", " ".join(cmd))
        return deployment
    logger.info("Deploying baseline %s ...", deployment)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Baseline deploy failed: %s", res.stderr.strip())
        return None
    logger.info("Baseline deployed: %s", deployment)
    return deployment


def deploy_fine_tuned(fine_tuned_model: str, dry_run: bool) -> str | None:
    deployment = "gpt-41-nano-student-v1"
    cmd = [
        _az(), "cognitiveservices", "account", "deployment", "create",
        "--name", FOUNDRY_ACCOUNT, "--resource-group", FOUNDRY_RG,
        "--deployment-name", deployment,
        "--model-name", fine_tuned_model, "--model-version", "1",
        "--model-format", "OpenAI",
        "--sku-name", "GlobalStandard", "--sku-capacity", "50",
    ]
    if dry_run:
        logger.info("[dry-run] would deploy student-v1: %s", " ".join(cmd))
        return deployment
    logger.info("Deploying student-v1 %s ...", deployment)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("student-v1 deploy failed: %s", res.stderr.strip())
        return None
    logger.info("student-v1 deployed: %s", deployment)
    return deployment


def baseline_eval(baseline_deployment: str, dry_run: bool) -> None:
    eval_run = f"dev3-baseline-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_real_integration.py"),
        "--eval-pool-jsonl", str(EVAL_POOL),
        "--baseline-deployment", baseline_deployment,
        "--student-deployment", baseline_deployment,  # base vs itself = pure baseline number
        "--eval-run-name", eval_run,
        "--onelake-workspace", ONELAKE_WORKSPACE,
    ]
    if dry_run:
        logger.info("[dry-run] would baseline-eval: %s", " ".join(cmd))
        return
    logger.info("Running baseline eval: %s", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed the dev3 continuous retraining loop")
    ap.add_argument("--epochs", type=int, default=2, help="epochs for the seed fine-tune")
    ap.add_argument("--training-file", default=str(INITIAL_SFT),
                    help="initial SFT set for student-v1 (default: tool-sft safe set)")
    ap.add_argument("--deploy-baseline", action="store_true",
                    help="deploy the base gpt-4.1-nano chat deployment")
    ap.add_argument("--baseline-eval", action="store_true",
                    help="run a baseline eval of the base model (implies --deploy-baseline)")
    ap.add_argument("--base-version", default="2025-04-14",
                    help="model version of the base model")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    base_model = settings.base_finetune_model
    training_file = Path(args.training_file)
    if not training_file.exists():
        raise FileNotFoundError(training_file)

    logger.info("=== dev3 bootstrap ===")
    logger.info("Foundry resource : %s / %s", FOUNDRY_ACCOUNT, FOUNDRY_RG)
    logger.info("Seed base model  : %s (continuous fine-tuning capable)", base_model)
    logger.info("Initial SFT file : %s", training_file)
    logger.info("Training type    : %s", settings.finetune_training_type or "default")

    state = _load_state()
    if state.get("current_student_model"):
        logger.warning("Loop already seeded (current_student_model=%s). "
                       "Delete %s to re-seed.", state["current_student_model"], STATE_FILE)
        return

    # 1/2. optional baseline deploy + eval
    if args.deploy_baseline or args.baseline_eval:
        baseline_dep = deploy_baseline(base_model, args.base_version, args.dry_run)
        if args.baseline_eval and baseline_dep:
            baseline_eval(baseline_dep, args.dry_run)

    # 3. seed fine-tune: student-v1 from the base model
    if args.dry_run:
        logger.info("[dry-run] would submit seed fine-tune: base=%s file=%s epochs=%d",
                    base_model, training_file, args.epochs)
        logger.info("[dry-run] would then set current_student_model in %s", STATE_FILE)
        return

    job = submit_finetune(
        training_file=str(training_file),
        model=base_model,
        suffix="dev3-seed-v1",
        epochs=args.epochs,
        training_type=settings.finetune_training_type,
        watch=True,
    )
    fine_tuned = getattr(job, "fine_tuned_model", None)
    if getattr(job, "status", None) != "succeeded" or not fine_tuned:
        logger.error("Seed fine-tune did not succeed: status=%s", getattr(job, "status", None))
        return

    deployed = deploy_fine_tuned(fine_tuned, args.dry_run)

    # 4. record continuation pointer
    state["current_student_model"] = fine_tuned
    state["history"].append({
        "event": "bootstrap-seed",
        "base_model": base_model,
        "fine_tuned_model": fine_tuned,
        "deployment": deployed,
        "epochs": args.epochs,
        "seeded_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_state(state)

    logger.info("=== Seed complete ===")
    logger.info("student-v1: %s", fine_tuned)
    if deployed:
        logger.info("student-v1 deployment: %s", deployed)
    logger.info("Continuation pointer set. The loop will now build on this model:")
    logger.info("    python scripts/run_retrain_loop.py --once")
    logger.info("Set STUDENT_FINETUNED_DEPLOYMENT in .env to the deployment name after validation.")


if __name__ == "__main__":
    main()
