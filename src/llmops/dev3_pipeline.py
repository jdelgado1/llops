"""Reusable helpers for the dev3 teacher-gold/text-SFT retraining loop.

This module captures the path that actually worked in the demo:
- Read Blu's retrain dataset from OneLake foundry_exports/<version>/train.jsonl.
- Convert top-level tool_calls to schema-free <tool_call> text SFT records.
- Upload training JSONL through the Azure OpenAI account endpoint.
- Submit GlobalStandard fine-tuning using the previous fine-tuned model as base.
- Deploy the resulting fine-tuned model with Azure CLI.

It intentionally avoids native tool schemas in the fine-tune training file because
Azure OpenAI preprocessing rejected those schemas for gpt-4.1-nano GlobalStandard
fine-tuning.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

ONELAKE_ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
AOAI_API_VERSION = "2025-04-01-preview"
TERMINAL = {"succeeded", "failed", "cancelled"}


def credential() -> DefaultAzureCredential:
    return DefaultAzureCredential(
        managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID") or None,
        exclude_interactive_browser_credential=True,
    )


def token(scope: str, cred: DefaultAzureCredential | None = None) -> str:
    return (cred or credential()).get_token(scope).token


def onelake_fs(workspace: str, cred: DefaultAzureCredential | None = None):
    return DataLakeServiceClient(
        account_url=ONELAKE_ACCOUNT_URL,
        credential=cred or credential(),
    ).get_file_system_client(workspace)


def download_onelake_jsonl(workspace: str, lakehouse: str, relative_path: str, dest: Path) -> list[dict[str, Any]]:
    fs = onelake_fs(workspace)
    full = f"{lakehouse}.Lakehouse/{relative_path}"
    data = fs.get_file_client(full).download_file().readall()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def text_tool_call(call: dict[str, Any]) -> str:
    payload = {"name": call["name"], "arguments": call.get("arguments") or {}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def convert_blu_rows_to_text_sft(rows: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt_messages = [m for m in row.get("messages", []) if m.get("role") != "assistant"]
        if not prompt_messages:
            raise ValueError(f"row {index} missing prompt messages")
        tool_calls = row.get("tool_calls", [])
        if not tool_calls:
            raise ValueError(f"row {index} missing tool_calls")
        converted.append({
            "messages": [
                *prompt_messages,
                {"role": "assistant", "content": "\n".join(text_tool_call(call) for call in tool_calls)},
            ]
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
        encoding="utf-8",
    )
    return converted


def aoai_urls(account: str) -> tuple[str, str]:
    account = account.lower()
    base = f"https://{account}.openai.azure.com/openai"
    return (
        f"{base}/files?api-version={AOAI_API_VERSION}",
        f"{base}/fine_tuning/jobs?api-version={AOAI_API_VERSION}",
    )


def upload_aoai_file(account: str, path: Path, cred: DefaultAzureCredential | None = None) -> str:
    files_url, _ = aoai_urls(account)
    active_cred = cred or credential()
    with path.open("rb") as handle:
        response = requests.post(
            files_url,
            headers={"Authorization": f"Bearer {token('https://cognitiveservices.azure.com/.default', active_cred)}"},
            data={"purpose": "fine-tune"},
            files={"file": (path.name, handle, "application/jsonl")},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()["id"]


def submit_global_finetune(
    account: str,
    base_model: str,
    training_file_id: str,
    suffix: str,
    epochs: int = 2,
    validation_file_id: str | None = None,
    cred: DefaultAzureCredential | None = None,
) -> dict[str, Any]:
    _, jobs_url = aoai_urls(account)
    body: dict[str, Any] = {
        "model": base_model,
        "training_file": training_file_id,
        "suffix": suffix,
        "method": {"type": "supervised"},
        "hyperparameters": {"n_epochs": epochs},
        "trainingType": "GlobalStandard",
    }
    if validation_file_id:
        body["validation_file"] = validation_file_id
    active_cred = cred or credential()
    response = requests.post(
        jobs_url,
        headers={
            "Authorization": f"Bearer {token('https://cognitiveservices.azure.com/.default', active_cred)}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def get_finetune_job(account: str, job_id: str, cred: DefaultAzureCredential | None = None) -> dict[str, Any]:
    account = account.lower()
    url = f"https://{account}.openai.azure.com/openai/fine_tuning/jobs/{job_id}?api-version={AOAI_API_VERSION}"
    active_cred = cred or credential()
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token('https://cognitiveservices.azure.com/.default', active_cred)}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def wait_for_finetune(account: str, job_id: str, poll_seconds: int = 60) -> dict[str, Any]:
    while True:
        job = get_finetune_job(account, job_id)
        print(f"fine-tune {job_id}: status={job.get('status')} error={job.get('error')}", flush=True)
        if job.get("status") in TERMINAL:
            return job
        time.sleep(poll_seconds)


def deploy_finetuned_model(
    account: str,
    resource_group: str,
    deployment_name: str,
    fine_tuned_model: str,
    capacity: int = 50,
) -> None:
    az = shutil.which("az") or shutil.which("az.cmd") or "az"
    command = [
        az,
        "cognitiveservices", "account", "deployment", "create",
        "--name", account,
        "--resource-group", resource_group,
        "--deployment-name", deployment_name,
        "--model-name", fine_tuned_model,
        "--model-version", "1",
        "--model-format", "OpenAI",
        "--sku-name", "GlobalStandard",
        "--sku-capacity", str(capacity),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Deployment failed: {result.stderr.strip()}")
    print(result.stdout.strip(), flush=True)
