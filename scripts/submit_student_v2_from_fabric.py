#!/usr/bin/env python
"""Submit student-v2 continuation fine-tune from Blu's OneLake export.

Reads:
  Files/llmops/foundry_exports/dev3-student-v2/train.jsonl

Converts Blu's candidate format:
  messages + top-level tool_calls

to the schema-free text SFT format that Azure OpenAI preprocessing accepted:
  messages: system, user, assistant(content=<tool_call>...</tool_call>)

Then submits a GlobalStandard fine-tune using the current student-v1 fine-tuned
model as the base.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

from llmops.config import get_settings
from llmops.models import get_client

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
VERSION = "dev3-student-v2"
REMOTE_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/foundry_exports/{VERSION}"
LOCAL_DIR = Path("artifacts") / VERSION
TRAIN_LOCAL = LOCAL_DIR / "train.jsonl"
TEXT_TRAIN_LOCAL = LOCAL_DIR / "train_text.jsonl"
JOB_OUT = LOCAL_DIR / "fine_tune_job.json"

BASE_MODEL = "gpt-4.1-nano-2025-04-14.ft-56718845ff1c4fd1a68a50e7c7800f6d-dev3-text-account-v1"
SUFFIX = "dev3-student-v2"
EPOCHS = 2
API_VERSION = "2025-04-01-preview"
ACCOUNT = "tunesefoundry"
JOBS_URL = f"https://{ACCOUNT}.openai.azure.com/openai/fine_tuning/jobs?api-version={API_VERSION}"
FILES_URL = f"https://{ACCOUNT}.openai.azure.com/openai/files?api-version={API_VERSION}"


def token(credential: AzureCliCredential) -> str:
    return credential.get_token("https://cognitiveservices.azure.com/.default").token


def download_from_onelake(credential: AzureCliCredential) -> list[dict[str, Any]]:
    fs = DataLakeServiceClient(
        account_url="https://onelake.dfs.fabric.microsoft.com",
        credential=credential,
    ).get_file_system_client(WORKSPACE)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    data = fs.get_file_client(f"{REMOTE_DIR}/train.jsonl").download_file().readall()
    TRAIN_LOCAL.write_bytes(data)
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    print(f"Downloaded {len(rows)} rows -> {TRAIN_LOCAL}", flush=True)
    return rows


def text_call(call: dict[str, Any]) -> str:
    payload = {"name": call["name"], "arguments": call.get("arguments") or {}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def convert_to_text_sft(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        prompt_messages = [m for m in row.get("messages", []) if m.get("role") != "assistant"]
        if not prompt_messages:
            raise ValueError("row missing prompt messages")
        content = "\n".join(text_call(call) for call in row.get("tool_calls", []))
        if not content:
            raise ValueError("row missing tool_calls")
        converted.append({
            "messages": [
                *prompt_messages,
                {"role": "assistant", "content": content},
            ]
        })
    TEXT_TRAIN_LOCAL.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
        encoding="utf-8",
    )
    print(f"Converted {len(converted)} rows -> {TEXT_TRAIN_LOCAL}", flush=True)
    return converted


def upload_file(path: Path, credential: AzureCliCredential) -> str:
    print(f"Uploading {path} to Azure OpenAI account endpoint...", flush=True)
    with path.open("rb") as handle:
        resp = requests.post(
            FILES_URL,
            headers={"Authorization": f"Bearer {token(credential)}"},
            data={"purpose": "fine-tune"},
            files={"file": (path.name, handle, "application/jsonl")},
            timeout=120,
        )
    print(f"upload status {resp.status_code}", flush=True)
    print(resp.text[:500], flush=True)
    resp.raise_for_status()
    return resp.json()["id"]


def submit_job(training_file_id: str, credential: AzureCliCredential) -> dict[str, Any]:
    body = {
        "model": BASE_MODEL,
        "training_file": training_file_id,
        "suffix": SUFFIX,
        "method": {"type": "supervised"},
        "hyperparameters": {"n_epochs": EPOCHS},
        "trainingType": "GlobalStandard",
    }
    (LOCAL_DIR / "fine_tune_request.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"Submitting continuation fine-tune from base model:\n  {BASE_MODEL}", flush=True)
    resp = requests.post(
        JOBS_URL,
        headers={"Authorization": f"Bearer {token(credential)}", "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    print(f"submit status {resp.status_code}", flush=True)
    print(resp.text, flush=True)
    resp.raise_for_status()
    job = resp.json()
    JOB_OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job


def poll(job: dict[str, Any], credential: AzureCliCredential) -> dict[str, Any]:
    status_url = f"https://{ACCOUNT}.openai.azure.com/openai/fine_tuning/jobs/{job['id']}?api-version={API_VERSION}"
    while job.get("status") not in {"succeeded", "failed", "cancelled"}:
        time.sleep(60)
        try:
            resp = requests.get(status_url, headers={"Authorization": f"Bearer {token(credential)}"}, timeout=60)
            resp.raise_for_status()
            job = resp.json()
            JOB_OUT.write_text(json.dumps(job, indent=2), encoding="utf-8")
            print(f"status: {job.get('status')} error={job.get('error')}", flush=True)
        except requests.RequestException as exc:
            print(f"transient status error: {exc}", flush=True)
    return job


def main() -> None:
    credential = AzureCliCredential()
    rows = download_from_onelake(credential)
    convert_to_text_sft(rows)
    training_file_id = upload_file(TEXT_TRAIN_LOCAL, credential)
    job = submit_job(training_file_id, credential)
    print(f"job id: {job['id']} status={job['status']}", flush=True)
    # Do not block forever inside ACA later, but locally we can watch.
    job = poll(job, credential)
    print(json.dumps(job, indent=2), flush=True)


if __name__ == "__main__":
    main()
