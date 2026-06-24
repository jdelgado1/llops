#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential

BASE = "https://tunesefoundry.openai.azure.com/openai/fine_tuning/jobs"
TRAIN = "file-7224defcef4c49a4a7b898be48f7763f"
VAL = "file-a78c92c9882a4b6bb2dbade56ac005d2"
VERSIONS = ["2026-03-01-preview", "2026-01-01-preview", "2025-10-01-preview", "2025-04-01-preview"]
FIELDS = ["training_type", "trainingType"]
VALUES = ["Global", "global", "GlobalStandard", "globalstandard", "global_standard"]

token = DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default").token
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

base_body = {
    "model": "gpt-4.1-nano",
    "training_file": TRAIN,
    "validation_file": VAL,
    "suffix": "dev3-teacher-gold-v1",
    "method": {"type": "supervised"},
    "hyperparameters": {"n_epochs": 2},
}

for version in VERSIONS:
    url = f"{BASE}?api-version={version}"
    for field in FIELDS:
        for value in VALUES:
            body = dict(base_body)
            body[field] = value
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            text = resp.text[:300].replace("\n", " ")
            print(f"{version} {field}={value} -> {resp.status_code}: {text}")
            if resp.status_code < 300:
                Path("artifacts/finetune_submit_success.json").write_text(resp.text, encoding="utf-8")
                raise SystemExit(0)
print("NO_SUCCESS")
