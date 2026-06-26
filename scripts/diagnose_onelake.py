#!/usr/bin/env python
"""Diagnostic: list OneLake workspaces and lakehouses accessible with current creds."""

import logging
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.WARNING)

ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
TARGET_WORKSPACE = "Fine Tune Demo"


def main() -> None:
    cred = DefaultAzureCredential()
    svc = DataLakeServiceClient(account_url=ACCOUNT_URL, credential=cred)

    print("=" * 60)
    print("OneLake workspaces (file systems) accessible:")
    print("=" * 60)
    workspaces = []
    try:
        for fs in svc.list_file_systems():
            workspaces.append(fs.name)
            print(f"  - {fs.name}")
    except Exception as e:
        print(f"  ⚠️  Could not list file systems: {e}")

    if not workspaces:
        print("  (none returned — listing may be restricted; trying target directly)")

    print()
    print("=" * 60)
    print(f"Items (lakehouses/warehouses) in workspace '{TARGET_WORKSPACE}':")
    print("=" * 60)
    try:
        fs_client = svc.get_file_system_client(TARGET_WORKSPACE)
        for path in fs_client.get_paths(recursive=False):
            kind = "DIR " if path.is_directory else "FILE"
            print(f"  [{kind}] {path.name}")
    except Exception as e:
        print(f"  ⚠️  Could not list items in '{TARGET_WORKSPACE}': {e}")


if __name__ == "__main__":
    main()
