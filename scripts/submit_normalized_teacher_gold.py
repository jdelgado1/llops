#!/usr/bin/env python
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential

from llmops.config import get_settings
from llmops.models import get_client

TRAIN = Path("artifacts/dev3-teacher-gold-small-normalized/train.jsonl")
VAL = Path("artifacts/dev3-teacher-gold-small-normalized/eval.jsonl")
MODEL = "gpt-4.1-nano"
SUFFIX = "dev3-teacher-gold-v2"
EPOCHS = 2
URL = "https://tunesefoundry.openai.azure.com/openai/fine_tuning/jobs?api-version=2025-04-01-preview"
OUT = Path("artifacts/normalized_teacher_gold_job.json")

print(f"Uploading normalized train file: {TRAIN}", flush=True)
client = get_client(get_settings())
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
    "hyperparameters": {"n_epochs": EPOCHS},
    "trainingType": "GlobalStandard",
}
Path("artifacts/normalized_teacher_gold_request.json").write_text(json.dumps(body, indent=2), encoding="utf-8")

credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default").token
print("Submitting fine-tune job via AOAI preview endpoint...", flush=True)
resp = requests.post(
    URL,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=body,
    timeout=120,
)
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
        token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        status = requests.get(status_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        status.raise_for_status()
        job = status.json()
        OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
        print(f"status: {job.get('status')}", flush=True)
    except requests.RequestException as exc:
        print(f"status check transient error: {exc}", flush=True)
        continue

print(json.dumps(job, indent=2), flush=True)
