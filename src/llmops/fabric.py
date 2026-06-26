"""Push production-grade traces from Foundry Tracing into Microsoft Fabric.

In production the golden/drift dataset of record lives in **Microsoft Fabric**
(Blu owns the DB + dashboards). The loop is:

    Foundry Hosted Agent  ->  Foundry Tracing (every tool-call turn)
        ->  AST-validate each traced call (keep correct vs flag rejected)
        ->  PUSH accepted traces to a Fabric Lakehouse table  (this module)
        ->  distill on accepted rows  ->  eval vs golden+drift  ->  promotion gate

Fabric is a **separate permission plane** (workspace + capacity + license) from
Azure RBAC. OneLake exposes an ADLS Gen2-compatible endpoint, so we upload with
``DefaultAzureCredential`` to:

    https://onelake.dfs.fabric.microsoft.com/<workspace>/<lakehouse>.Lakehouse/Files/<path>

If the Fabric env/SDK isn't configured we still write a **Fabric-ready** JSONL
locally and print the exact upload command, so the step is never a hard blocker.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .ast_check import check_ast
from .config import get_settings

ARTIFACTS = Path("artifacts")
ONELAKE_HOST = "https://onelake.dfs.fabric.microsoft.com"


def _accepted_rows(traces_path: Path) -> list[dict]:
    """Normalize traces to the Fabric golden/drift table schema; keep AST-correct."""
    rows: list[dict] = []
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            calls = t.get("tool_calls") or []
            ref = t.get("reference") or []
            # If a reference is present, only keep AST-correct rows; otherwise keep all
            # (live Foundry traces may have no reference — accept and tag for review).
            accepted = True
            reason = "no_reference"
            if ref:
                v = check_ast(calls, ref, t.get("tools") or [])
                accepted, reason = v.correct, v.reason
            if not accepted:
                continue
            rows.append(
                {
                    "tid": t.get("tid"),
                    "category": t.get("category"),
                    "request": next(
                        (m.get("content") for m in t.get("messages", []) if m.get("role") == "user"),
                        "",
                    ),
                    "tool_calls": json.dumps(calls, ensure_ascii=False),
                    "accepted": accepted,
                    "accept_reason": reason,
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
    return rows


def _try_upload_onelake(local_file: Path, workspace: str, lakehouse: str) -> str | None:
    """Upload to OneLake via the ADLS Gen2 API. Returns the URL or None if unavailable."""
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.filedatalake import DataLakeServiceClient
    except ImportError:
        return None

    file_path = f"{lakehouse}.Lakehouse/Files/golden_traces/{local_file.name}"
    service = DataLakeServiceClient(
        account_url=ONELAKE_HOST, credential=DefaultAzureCredential()
    )
    fs = service.get_file_system_client(workspace)
    fc = fs.get_file_client(file_path)
    fc.upload_data(local_file.read_bytes(), overwrite=True)
    return f"{ONELAKE_HOST}/{workspace}/{file_path}"


def push(traces_path: str, out_path: str | None = None) -> Path:
    settings = get_settings()
    src = Path(traces_path)
    if not src.exists():
        raise FileNotFoundError(src)

    rows = _accepted_rows(src)
    if not rows:
        raise RuntimeError(f"No accepted rows in {src} to push.")

    ARTIFACTS.mkdir(exist_ok=True)
    out = Path(out_path) if out_path else ARTIFACTS / f"fabric-golden-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prepared {len(rows)} accepted rows (Fabric golden/drift schema) -> {out}")

    ws, lh = settings.fabric_workspace_id, settings.fabric_lakehouse
    if ws and lh:
        url = _try_upload_onelake(out, ws, lh)
        if url:
            print(f"Uploaded to OneLake -> {url}")
        else:
            print(
                "Fabric env set but azure-storage-file-datalake not installed.\n"
                "  pip install azure-storage-file-datalake\n"
                "then re-run, or upload manually with:\n"
                f"  az storage fs file upload --account-name onelake "
                f"--file-system {ws} "
                f"--path '{lh}.Lakehouse/Files/golden_traces/{out.name}' "
                f"--source {out} --auth-mode login"
            )
    else:
        print(
            "FABRIC_WORKSPACE_ID / FABRIC_LAKEHOUSE not set — wrote Fabric-ready file only.\n"
            "Set them in .env to auto-push to OneLake, or hand this file to Blu for ingestion."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Push accepted traces from Foundry into Microsoft Fabric (OneLake).")
    ap.add_argument("--traces", required=True, help="path to tool-call traces JSONL (with optional reference)")
    ap.add_argument("--out", default=None, help="local Fabric-ready JSONL output path")
    args = ap.parse_args()
    push(args.traces, args.out)


if __name__ == "__main__":
    main()
