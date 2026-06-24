#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "ftjob-c6acb7b3e068422fa372b3c039c45b85"
URL = f"https://tunesefoundry.openai.azure.com/openai/fine_tuning/jobs/{JOB_ID}?api-version=2025-04-01-preview"
OUT = Path(f"artifacts/{JOB_ID}.json")
TERMINAL = {"succeeded", "failed", "cancelled"}
cred = DefaultAzureCredential()

while True:
    try:
        token = cred.get_token("https://cognitiveservices.azure.com/.default").token
        r = requests.get(URL, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        r.raise_for_status()
        job = r.json()
        OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
        print(f"{time.strftime('%H:%M:%S')} status={job.get('status')} fine_tuned_model={job.get('fine_tuned_model')} error={job.get('error')}", flush=True)
        if job.get("status") in TERMINAL:
            break
    except requests.RequestException as exc:
        print(f"{time.strftime('%H:%M:%S')} transient status error: {exc}", flush=True)
    time.sleep(60)
