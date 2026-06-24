#!/usr/bin/env python
"""Create Lakehouse tables for the executive reporting dashboard.

Only touches reporting files/tables:
  Files/llmops/reporting_csv/*.csv
  Tables/executive_scorecard_latest
  Tables/executive_kpis_latest
  Tables/executive_scorecard_history
  Tables/promotion_decisions

Uses Fabric Lakehouse Load Table API, which supports CSV/Parquet but not JSONL.
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import Any

import requests
from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

WORKSPACE_NAME = "Fine Tune Demo"
LAKEHOUSE_NAME = "lh_llmops"
WORKSPACE_ID = "dabb8e60-9286-452f-9d84-c477a7781999"
LAKEHOUSE_ID = "6611a819-3e9b-4156-9251-c18f89045684"
ONELAKE_ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
REPORTING_DIR = f"{LAKEHOUSE_NAME}.Lakehouse/Files/llmops/reporting"
REPORTING_CSV_DIR = f"{LAKEHOUSE_NAME}.Lakehouse/Files/llmops/reporting_csv"
FABRIC_API = "https://api.fabric.microsoft.com/v1"

TABLES = {
    "executive_scorecard_latest": "executive_scorecard_latest.jsonl",
    "executive_kpis_latest": "executive_kpis_latest.jsonl",
    "executive_scorecard_history": "executive_scorecard_history.jsonl",
    "promotion_decisions": "promotion_decisions.jsonl",
}


def fabric_token(cred: AzureCliCredential) -> str:
    return cred.get_token("https://api.fabric.microsoft.com/.default").token


def read_jsonl(fs, filename: str) -> list[dict[str, Any]]:
    path = f"{REPORTING_DIR}/{filename}"
    raw = fs.get_file_client(path).download_file().readall().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot convert empty rows to CSV")
    fields = sorted({key for row in rows for key in row.keys()})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        clean = {}
        for key in fields:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                clean[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = value
        writer.writerow(clean)
    return output.getvalue().encode("utf-8")


def upload_csv(fs, table_name: str, content: bytes) -> str:
    relative = f"Files/llmops/reporting_csv/{table_name}.csv"
    full = f"{LAKEHOUSE_NAME}.Lakehouse/{relative}"
    fs.get_directory_client("/".join(full.split("/")[:-1])).create_directory()
    fs.get_file_client(full).upload_data(content, overwrite=True)
    print(f"Uploaded CSV -> {relative}")
    return relative


def load_table(cred: AzureCliCredential, table_name: str, relative_path: str) -> None:
    url = f"{FABRIC_API}/workspaces/{WORKSPACE_ID}/lakehouses/{LAKEHOUSE_ID}/tables/{table_name}/load"
    body = {
        "relativePath": relative_path,
        "pathType": "File",
        "mode": "Overwrite",
        "recursive": False,
        "formatOptions": {
            "format": "Csv",
            "header": True,
            "delimiter": ",",
        },
    }
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {fabric_token(cred)}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    print(f"Load table {table_name}: {response.status_code}")
    if response.status_code >= 400:
        print(response.text)
        response.raise_for_status()
    operation_url = response.headers.get("Location")
    if not operation_url:
        return
    while True:
        time.sleep(int(response.headers.get("Retry-After", "5")))
        poll = requests.get(operation_url, headers={"Authorization": f"Bearer {fabric_token(cred)}"}, timeout=60)
        if poll.status_code >= 400:
            print(poll.text)
            poll.raise_for_status()
        payload = poll.json()
        status = payload.get("status")
        print(f"  {table_name} operation status: {status}")
        if status in {"Succeeded", "Failed", "Cancelled"}:
            if status != "Succeeded":
                raise RuntimeError(f"Load table {table_name} ended with {status}: {payload}")
            return


def main() -> None:
    cred = AzureCliCredential()
    fs = DataLakeServiceClient(account_url=ONELAKE_ACCOUNT_URL, credential=cred).get_file_system_client(WORKSPACE_NAME)
    for table_name, filename in TABLES.items():
        rows = read_jsonl(fs, filename)
        csv_bytes = rows_to_csv(rows)
        relative_path = upload_csv(fs, table_name, csv_bytes)
        load_table(cred, table_name, relative_path)
    print("Reporting Lakehouse tables are ready.")


if __name__ == "__main__":
    main()
