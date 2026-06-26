#!/usr/bin/env python
from __future__ import annotations

import csv
import io
import json
from typing import Any

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

WORKSPACE = "Fine Tune Demo"
LAKEHOUSE = "lh_llmops"
ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
REPORTING_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/reporting"
REPORTING_CSV_DIR = f"{LAKEHOUSE}.Lakehouse/Files/llmops/reporting_csv"

FILES = {
    "executive_scorecard_latest": "executive_scorecard_latest.jsonl",
    "executive_kpis_latest": "executive_kpis_latest.jsonl",
    "executive_scorecard_history": "executive_scorecard_history.jsonl",
    "promotion_decisions": "promotion_decisions.jsonl",
}


def rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    fields = sorted({key for row in rows for key in row.keys()})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
            else "" if value is None
            else value
            for key, value in row.items()
        })
    return output.getvalue().encode("utf-8")


def main() -> None:
    fs = DataLakeServiceClient(account_url=ACCOUNT_URL, credential=AzureCliCredential()).get_file_system_client(WORKSPACE)
    fs.get_directory_client(REPORTING_CSV_DIR).create_directory()
    for table, filename in FILES.items():
        src = f"{REPORTING_DIR}/{filename}"
        raw = fs.get_file_client(src).download_file().readall().decode("utf-8")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        dest = f"{REPORTING_CSV_DIR}/{table}.csv"
        fs.get_file_client(dest).upload_data(rows_to_csv(rows), overwrite=True)
        print(f"Uploaded {dest} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
