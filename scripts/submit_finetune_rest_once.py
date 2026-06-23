#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential

URL = "https://tunesefoundry.services.ai.azure.com/api/projects/proj-default/openai/v1/fine_tuning/jobs"
RESPONSE = Path("artifacts/last_finetune_response.txt")

body = {
    "model": "gpt-4o-mini",
    "training_file": "file-c53ced9d80d74e599f878568d4c94044",
    "suffix": "dev3-seed-v1",
    "method": {"type": "supervised"},
    "hyperparameters": {"n_epochs": 2},
}
Path("artifacts/last_finetune_request.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
token = DefaultAzureCredential().get_token("https://ai.azure.com/.default").token
resp = requests.post(
    URL,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=body,
    timeout=120,
)
print("STATUS", resp.status_code)
print(resp.text)
RESPONSE.write_text(resp.text, encoding="utf-8")
resp.raise_for_status()
