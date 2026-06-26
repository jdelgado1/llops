#!/usr/bin/env python
"""Check whether Blu's retrain-v2 dataset is present in OneLake foundry_exports."""

import logging
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.WARNING)

ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
WORKSPACE = "Fine Tune Demo"
EXPORTS_DIR = "lh_llmops.Lakehouse/Files/llmops/foundry_exports"


def main() -> None:
    cred = DefaultAzureCredential()
    svc = DataLakeServiceClient(account_url=ACCOUNT_URL, credential=cred)
    fs = svc.get_file_system_client(WORKSPACE)

    print("=" * 60)
    print(f"Contents of {EXPORTS_DIR}:")
    print("=" * 60)
    try:
        for path in fs.get_paths(path=EXPORTS_DIR, recursive=True):
            kind = "DIR " if path.is_directory else "FILE"
            size = "" if path.is_directory else f"  ({path.content_length} bytes)"
            rel = path.name.replace(f"{EXPORTS_DIR}/", "")
            print(f"  [{kind}] {rel}{size}")
    except Exception as e:
        print(f"  ⚠️  {e}")


if __name__ == "__main__":
    main()
