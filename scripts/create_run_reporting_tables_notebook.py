#!/usr/bin/env python
from __future__ import annotations

import base64
import json
import time

import requests
from azure.identity import AzureCliCredential

WORKSPACE_ID = "dabb8e60-9286-452f-9d84-c477a7781999"
LAKEHOUSE_ID = "6611a819-3e9b-4156-9251-c18f89045684"
LAKEHOUSE_NAME = "lh_llmops"
NOTEBOOK_NAME = "99_create_executive_reporting_tables"
FABRIC = "https://api.fabric.microsoft.com/v1"

NOTEBOOK_CODE = f'''# Fabric notebook source

# METADATA ********************

# META {{
# META   "kernel_info": {{
# META     "name": "synapse_pyspark"
# META   }},
# META   "dependencies": {{
# META     "lakehouse": {{
# META       "default_lakehouse": "{LAKEHOUSE_ID}",
# META       "default_lakehouse_name": "{LAKEHOUSE_NAME}",
# META       "default_lakehouse_workspace_id": "{WORKSPACE_ID}",
# META       "known_lakehouses": [
# META         {{
# META           "id": "{LAKEHOUSE_ID}"
# META         }}
# META       ]
# META     }}
# META   }}
# META }}

# CELL ********************

# Create executive dashboard Lakehouse tables from reporting CSV files.
# Scope: reporting-only. Does not touch raw evals/traces/foundry_exports.

from pyspark.sql import functions as F

TABLES = [
    "executive_scorecard_latest",
    "executive_kpis_latest",
    "executive_scorecard_history",
    "promotion_decisions",
]

for table in TABLES:
    path = f"Files/llmops/reporting_csv/{{table}}.csv"
    print(f"Loading {{path}} -> {{table}}")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .csv(path)
    )
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    print(table, spark.table(table).count())

display(spark.sql("SHOW TABLES"))

# CELL ********************
'''

PLATFORM = {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
    "metadata": {
        "type": "Notebook",
        "displayName": NOTEBOOK_NAME,
        "description": "Creates executive dashboard reporting tables from Files/llmops/reporting_csv"
    },
    "config": {
        "version": "2.0",
        "logicalId": "00000000-0000-0000-0000-000000000000"
    }
}


def token(cred: AzureCliCredential) -> str:
    return cred.get_token("https://api.fabric.microsoft.com/.default").token


def headers(cred: AzureCliCredential) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(cred)}", "Content-Type": "application/json"}


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def find_notebook(cred: AzureCliCredential) -> str | None:
    r = requests.get(f"{FABRIC}/workspaces/{WORKSPACE_ID}/items", headers=headers(cred), timeout=60)
    r.raise_for_status()
    for item in r.json().get("value", []):
        if item.get("type") == "Notebook" and item.get("displayName") == NOTEBOOK_NAME:
            return item["id"]
    return None


def create_notebook(cred: AzureCliCredential) -> str:
    existing = find_notebook(cred)
    if existing:
        print(f"Using existing notebook {NOTEBOOK_NAME}: {existing}")
        return existing
    body = {
        "displayName": NOTEBOOK_NAME,
        "type": "Notebook",
        "definition": {
            "format": "ipynb",
            "parts": [
                {"path": "notebook-content.py", "payload": b64(NOTEBOOK_CODE), "payloadType": "InlineBase64"},
                {"path": ".platform", "payload": b64(json.dumps(PLATFORM, indent=2)), "payloadType": "InlineBase64"},
            ],
        },
    }
    r = requests.post(f"{FABRIC}/workspaces/{WORKSPACE_ID}/items", headers=headers(cred), json=body, timeout=60)
    print("create notebook", r.status_code, r.text[:500])
    if r.status_code == 202:
        location = r.headers.get("Location")
        while location:
            time.sleep(int(r.headers.get("Retry-After", "10")))
            poll = requests.get(location, headers={"Authorization": f"Bearer {token(cred)}"}, timeout=60)
            poll.raise_for_status()
            payload = poll.json()
            print("create poll", payload.get("status"), payload.get("percentComplete"))
            if payload.get("status") in {"Succeeded", "Failed", "Cancelled"}:
                if payload.get("status") != "Succeeded":
                    raise RuntimeError(f"Notebook creation failed: {payload}")
                break
        found = find_notebook(cred)
        if not found:
            raise RuntimeError("Notebook creation completed but item was not found by displayName")
        return found
    r.raise_for_status()
    return r.json()["id"]


def run_notebook(cred: AzureCliCredential, notebook_id: str) -> None:
    url = f"{FABRIC}/workspaces/{WORKSPACE_ID}/items/{notebook_id}/jobs/DefaultJob/instances"
    r = requests.post(url, headers=headers(cred), timeout=60)
    print("run notebook", r.status_code)
    if r.status_code >= 400:
        print(r.text)
        r.raise_for_status()
    location = r.headers.get("Location")
    if not location:
        print("No Location header returned; run accepted without poll URL")
        return
    while True:
        time.sleep(int(r.headers.get("Retry-After", "30")))
        p = requests.get(location, headers={"Authorization": f"Bearer {token(cred)}"}, timeout=60)
        print("poll", p.status_code, p.text[:500])
        p.raise_for_status()
        payload = p.json()
        status = payload.get("status")
        if status in {"Succeeded", "Failed", "Cancelled"}:
            if status != "Succeeded":
                raise RuntimeError(f"Notebook run ended with {status}: {payload}")
            return


def main() -> None:
    cred = AzureCliCredential()
    notebook_id = create_notebook(cred)
    run_notebook(cred, notebook_id)
    print("Reporting table notebook completed.")


if __name__ == "__main__":
    main()
