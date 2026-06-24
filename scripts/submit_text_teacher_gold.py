#!/usr/bin/env python
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from azure.identity import AzureCliCredential

from llmops.config import get_settings
from llmops.models import get_client

TRAIN = Path("artifacts/dev3-teacher-gold-small-text/train.jsonl")
VAL = Path("artifacts/dev3-teacher-gold-small-text/eval.jsonl")
MODEL = "gpt-4.1-nano"
SUFFIX = "dev3-teacher-gold-text-v1"
URL = "https://tunesefoundry.openai.azure.com/openai/fine_tuning/jobs?api-version=2025-04-01-preview"
OUT = Path("artifacts/text_teacher_gold_job.json")

client = get_client(get_settings())
print(f"Uploading text train file: {TRAIN}", flush=True)
train_file = client.files.create(file=(TRAIN.name, TRAIN.read_bytes()), purpose="fine-tune")
print(f"  train file id: {train_file.id}", flush=True)
val_file = client.files.create(file=(VAL.name, VAL.read_bytes()), purpose="fine-tune")
print(f"  val file id: {val_file.id}", flush=True)

body = {
    "model": MODEL,
    "training_file": train_file.id,
    "validation_file": val_file.id,
    "suffix": SUFFIX,
    "method": {"type": "supervised"},
    "hyperparameters": {"n_epochs": 2},
    "trainingType": "GlobalStandard",
}
Path("artifacts/text_teacher_gold_request.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
cred = AzureCliCredential()
token = cred.get_token("https://cognitiveservices.azure.com/.default").token
resp = requests.post(URL, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=body, timeout=120)
print("STATUS", resp.status_code, flush=True)
print(resp.text, flush=True)
if resp.status_code >= 400:
    raise SystemExit(1)
job = resp.json()
OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
print(f"job id: {job['id']} status={job['status']}", flush=True)

status_url = f"https://tunesefoundry.openai.azure.com/openai/fine_tuning/jobs/{job['id']}?api-version=2025-04-01-preview"
while job.get("status") not in {"succeeded", "failed", "cancelled"}:
    time.sleep(60)
    try:
        token = cred.get_token("https://cognitiveservices.azure.com/.default").token
        r = requests.get(status_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        r.raise_for_status()
        job = r.json()
        OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
        print(f"status: {job.get('status')}", flush=True)
    except requests.RequestException as exc:
        print(f"status check transient error: {exc}", flush=True)

print(json.dumps(job, indent=2), flush=True)
