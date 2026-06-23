#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential

URL = "https://tunesefoundry.openai.azure.com/openai/fine_tuning/jobs?api-version=2025-04-01-preview"
RESPONSE = Path("artifacts/last_finetune_aoai_preview_response.txt")

body = {
    "Model": "gpt-4.1-nano",
    "TrainingFile": "file-c53ced9d80d74e599f878568d4c94044",
    "Suffix": "dev3-seed-v1",
    "Method": {"Type": "supervised"},
    "Hyperparameters": {"NEpochs": 2},
    "fineTuningJob": {"TrainingType": "Global"},
}
Path("artifacts/last_finetune_aoai_preview_request.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
token = DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default").token
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
