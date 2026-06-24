"""Submit a Foundry **managed SFT** (supervised fine-tune) job via the
OpenAI-compatible fine-tuning API — no GPU; the result deploys serverless.

This is the pipeline's "distill" step in code form:

    upload JSONL  ->  create fine-tuning job (base model + epochs)  ->  (watch)

Used for both fine-tunes in Option A:
  - the **format-primer** baseline (generic, off-distribution tool calls), and
  - the real **distillation** SFT (rejection-sampled teacher tool-call traces).

After the job reaches ``succeeded``, deploy the fine-tuned model (serverless) in
Foundry and put its deployment name in ``.env`` (``BASELINE_DEPLOYMENT`` for the
primer, ``STUDENT_FINETUNED_DEPLOYMENT`` for the distill).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import requests
from azure.identity import DefaultAzureCredential

from .config import get_settings
from .models import get_client

DEFAULT_BASE_MODEL = "Qwen3-32B"
TERMINAL = {"succeeded", "failed", "cancelled"}


def _aoai_jobs_url(job_id: str | None = None) -> str:
    account = (os.environ.get("FOUNDRY_AOAI_ACCOUNT") or "TuneSEfoundry").lower()
    base = f"https://{account}.openai.azure.com/openai/fine_tuning/jobs"
    if job_id:
        base = f"{base}/{job_id}"
    return f"{base}?api-version=2025-04-01-preview"


def _aoai_headers() -> dict[str, str]:
    token = DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default").token
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _as_job(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def _create_global_finetune_job(kwargs: dict, training_type: str | None) -> SimpleNamespace:
    # The project /openai/v1 endpoint currently ignores Global training type for
    # gpt-4.1-nano. The account preview endpoint accepts trainingType=GlobalStandard.
    body = {
        "model": kwargs["model"],
        "training_file": kwargs["training_file"],
        "suffix": kwargs.get("suffix"),
        "method": kwargs.get("method") or {"type": "supervised"},
        "hyperparameters": kwargs.get("hyperparameters", {}),
        "trainingType": "GlobalStandard" if training_type else None,
    }
    if kwargs.get("validation_file"):
        body["validation_file"] = kwargs["validation_file"]
    body = {k: v for k, v in body.items() if v is not None}

    resp = requests.post(_aoai_jobs_url(), headers=_aoai_headers(), json=body, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Global fine-tune submit failed: {resp.status_code} {resp.text}")
    return _as_job(resp.json())


def _retrieve_global_finetune_job(job_id: str) -> SimpleNamespace:
    resp = requests.get(_aoai_jobs_url(job_id), headers=_aoai_headers(), timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Global fine-tune status failed: {resp.status_code} {resp.text}")
    return _as_job(resp.json())


def submit_finetune(
    training_file: str,
    model: str = DEFAULT_BASE_MODEL,
    suffix: str | None = None,
    epochs: int | None = None,
    validation_file: str | None = None,
    training_type: str | None = "Global",
    watch: bool = False,
):
    """Upload the training file and create a managed SFT job. Returns the job."""
    settings = get_settings()
    client = get_client(settings)

    path = Path(training_file)
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Uploading training file: {path}")
    up = client.files.create(file=(path.name, path.read_bytes()), purpose="fine-tune")
    print(f"  training file id: {up.id}")

    kwargs: dict = {
        "training_file": up.id,
        "model": model,
        "method": {"type": "supervised"},
    }
    if suffix:
        kwargs["suffix"] = suffix
    if validation_file:
        vp = Path(validation_file)
        v = client.files.create(file=(vp.name, vp.read_bytes()), purpose="fine-tune")
        kwargs["validation_file"] = v.id
        print(f"  validation file id: {v.id}")
    if epochs:
        kwargs["hyperparameters"] = {"n_epochs": epochs}

    print(f"Creating managed SFT job on base model '{model}'"
          + (f", trainingType={training_type}" if training_type else "")
          + (f", {epochs} epochs" if epochs else "") + " ...")
    if training_type and training_type.lower() in {"global", "globalstandard", "global_standard"}:
        job = _create_global_finetune_job(kwargs, training_type)
    else:
        job = client.fine_tuning.jobs.create(**kwargs)
    print(f"  job id : {job.id}")
    print(f"  status : {job.status}")

    if watch:
        if training_type and training_type.lower() in {"global", "globalstandard", "global_standard"}:
            job = _watch_global(job.id)
        else:
            job = _watch(client, job.id)

    print("\nNext: when status=succeeded, deploy the fine-tuned model (serverless)")
    print("and set its deployment name in .env (BASELINE_DEPLOYMENT or STUDENT_FINETUNED_DEPLOYMENT).")
    return job


def status(job_id: str, watch: bool = False):
    """Print a fine-tune job's status (optionally poll until terminal)."""
    settings = get_settings()
    client = get_client(settings)
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"job {job.id}: status={job.status} model={getattr(job, 'model', '?')} "
          f"fine_tuned_model={getattr(job, 'fine_tuned_model', None)}")
    if watch and job.status not in TERMINAL:
        job = _watch(client, job_id)
    return job


def _watch(client, job_id: str, interval: int = 60):
    print("Watching (Ctrl-C to stop; the job keeps running) ...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  [{time.strftime('%H:%M:%S')}] status={job.status}")
        if job.status in TERMINAL:
            if getattr(job, "fine_tuned_model", None):
                print(f"  fine_tuned_model: {job.fine_tuned_model}")
            return job
        time.sleep(interval)


def _watch_global(job_id: str, interval: int = 60):
    print("Watching Global fine-tune (Ctrl-C to stop; the job keeps running) ...")
    while True:
        job = _retrieve_global_finetune_job(job_id)
        print(f"  [{time.strftime('%H:%M:%S')}] status={job.status}")
        if job.status in TERMINAL:
            if getattr(job, "fine_tuned_model", None):
                print(f"  fine_tuned_model: {job.fine_tuned_model}")
            return job
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit a Foundry managed SFT job (or check status).")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("submit", help="upload + create a fine-tune job")
    s.add_argument("--training-file", required=True)
    s.add_argument("--model", default=DEFAULT_BASE_MODEL, help="base model (default Qwen3-32B)")
    s.add_argument("--suffix", default=None, help="suffix for the fine-tuned model name")
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--validation-file", default=None)
    s.add_argument("--training-type", default="Global", help="Global (Qwen) | Standard")
    s.add_argument("--watch", action="store_true", help="poll until the job finishes")

    st = sub.add_parser("status", help="check a job's status")
    st.add_argument("--job-id", required=True)
    st.add_argument("--watch", action="store_true")

    args = ap.parse_args()
    if args.cmd == "status":
        status(args.job_id, watch=args.watch)
    else:  # default to submit
        submit_finetune(
            training_file=args.training_file,
            model=args.model,
            suffix=args.suffix,
            epochs=args.epochs,
            validation_file=args.validation_file,
            training_type=args.training_type,
            watch=args.watch,
        )


if __name__ == "__main__":
    main()
