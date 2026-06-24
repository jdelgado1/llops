#!/usr/bin/env python
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from azure.identity import AzureCliCredential

TRAIN = Path("artifacts/dev3-teacher-gold-small-text/train.jsonl")
VAL = Path("artifacts/dev3-teacher-gold-small-text/eval.jsonl")
API = "2025-04-01-preview"
BASE = "https://tunesefoundry.openai.azure.com/openai"
JOBS = f"{BASE}/fine_tuning/jobs?api-version={API}"
FILES = f"{BASE}/files?api-version={API}"
OUT = Path("artifacts/text_teacher_gold_account_job.json")
cred = AzureCliCredential()


def token() -> str:
    return cred.get_token("https://cognitiveservices.azure.com/.default").token


def upload(path: Path) -> str:
    print(f"Uploading via account endpoint: {path}", flush=True)
    with path.open("rb") as handle:
        resp = requests.post(
            FILES,
            headers={"Authorization": f"Bearer {token()}"},
            data={"purpose": "fine-tune"},
            files={"file": (path.name, handle, "application/jsonl")},
            timeout=120,
        )
    print("upload status", resp.status_code, flush=True)
    print(resp.text[:500], flush=True)
    resp.raise_for_status()
    return resp.json()["id"]


train_id = upload(TRAIN)
val_id = upload(VAL)
body = {
    "model": "gpt-4.1-nano",
    "training_file": train_id,
    "validation_file": val_id,
    "suffix": "dev3-text-account-v1",
    "method": {"type": "supervised"},
    "hyperparameters": {"n_epochs": 2},
    "trainingType": "GlobalStandard",
}
Path("artifacts/text_teacher_gold_account_request.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
print("Submitting job with account-scoped file IDs...", flush=True)
resp = requests.post(
    JOBS,
    headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"},
    json=body,
    timeout=120,
)
print("submit status", resp.status_code, flush=True)
print(resp.text, flush=True)
resp.raise_for_status()
job = resp.json()
OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
print(f"job id: {job['id']} status={job['status']}", flush=True)

status_url = f"{BASE}/fine_tuning/jobs/{job['id']}?api-version={API}"
while job.get("status") not in {"succeeded", "failed", "cancelled"}:
    time.sleep(60)
    try:
        r = requests.get(status_url, headers={"Authorization": f"Bearer {token()}"}, timeout=60)
        r.raise_for_status()
        job = r.json()
        OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
        print(f"status: {job.get('status')} error={job.get('error')}", flush=True)
    except requests.RequestException as exc:
        print(f"transient status error: {exc}", flush=True)

print(json.dumps(job, indent=2), flush=True)
