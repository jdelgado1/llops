"""
Fabric integration layer for TMG LLMOps distillation loop.
Exports Foundry traces & eval results to Fabric Lakehouse paths.
Handles Fabric -> Foundry dataset consumption.

Paths (in Lakehouse lh_llmops):
  - Files/llmops/raw/foundry_traces/        ← agent/model call traces (JSONL)
  - Files/llmops/raw/foundry_evals/         ← eval results by run (JSONL)
  - Files/llmops/foundry_exports/           ← curated datasets for Foundry (JSONL)
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# OneLake ADLS Gen2 endpoint
_ONELAKE_ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"


class FabricExporter:
    """Export Foundry traces and eval results to Fabric Lakehouse."""

    def __init__(
        self,
        lakehouse_root: str = "/Volumes/lh_llmops",
        debug: bool = False,
        onelake_workspace: Optional[str] = None,
        onelake_lakehouse: str = "lh_llmops",
    ):
        """
        Initialize Fabric exporter.

        Args:
            lakehouse_root: Local path fallback (used when onelake_workspace is not set)
            debug: If True, write to local ./fabric_export_debug/ instead
            onelake_workspace: Fabric workspace name (e.g. "Fine Tune Demo").
                               When set, files are uploaded directly to OneLake via REST API.
            onelake_lakehouse: Lakehouse name inside the workspace (default: lh_llmops)
        """
        self.debug = debug
        self.onelake_workspace = onelake_workspace
        self.onelake_lakehouse = onelake_lakehouse
        self._fs_client = None  # lazy-init

        if onelake_workspace:
            # OneLake mode — paths are virtual (used for logging only)
            self.lakehouse_root = Path(f"onelake://{onelake_workspace}/{onelake_lakehouse}.Lakehouse")
            logger.info(f"OneLake mode: workspace='{onelake_workspace}' lakehouse='{onelake_lakehouse}'")
        elif debug:
            self.lakehouse_root = Path("./fabric_export_debug")
            self.lakehouse_root.mkdir(exist_ok=True, parents=True)
        else:
            self.lakehouse_root = Path(lakehouse_root)

        self.traces_dir = self.lakehouse_root / "Files/llmops/raw/foundry_traces"
        self.evals_dir = self.lakehouse_root / "Files/llmops/raw/foundry_evals"
        self.exports_dir = self.lakehouse_root / "Files/llmops/foundry_exports"

        if not onelake_workspace:
            for d in [self.traces_dir, self.evals_dir, self.exports_dir]:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f"Fabric path ready: {d}")
        else:
            logger.info(f"OneLake paths: {self.traces_dir}, {self.evals_dir}, {self.exports_dir}")

    def _get_fs_client(self):
        """Lazy-init the OneLake DataLakeServiceClient."""
        if self._fs_client is None:
            from azure.storage.filedatalake import DataLakeServiceClient
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            service = DataLakeServiceClient(
                account_url=_ONELAKE_ACCOUNT_URL,
                credential=credential,
            )
            self._fs_client = service.get_file_system_client(self.onelake_workspace)
        return self._fs_client

    def _write_file(self, fabric_path: Path, content: bytes) -> None:
        """
        Write bytes to either OneLake (if configured) or local filesystem.

        fabric_path: full path under the lakehouse root, e.g.
                     onelake://Fine Tune Demo/lh_llmops.Lakehouse/Files/...
        """
        if self.onelake_workspace:
            # Strip the virtual prefix to get the path inside the file system
            # fabric_path looks like: onelake://<workspace>/<lakehouse>.Lakehouse/<rest>
            # OneLake ADLS path inside the file system = <lakehouse>.Lakehouse/<rest>
            relative = str(fabric_path).replace(
                f"onelake://{self.onelake_workspace}/", ""
            ).replace("\\", "/")
            fs = self._get_fs_client()
            dir_path = "/".join(relative.split("/")[:-1])
            fs.get_directory_client(dir_path).create_directory()
            file_client = fs.get_file_client(relative)
            file_client.upload_data(content, overwrite=True)
            logger.info(f"Uploaded to OneLake: {relative}")
        else:
            fabric_path.parent.mkdir(parents=True, exist_ok=True)
            fabric_path.write_bytes(content)

    def export_traces(
        self,
        traces: List[Dict[str, Any]],
        model_name: str,
        agent_name: str = "tmg_ops_agent",
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        Export Foundry Tracing records to Fabric.

        Each record: one agent/model call (request -> tool_calls).

        Args:
            traces: List of trace dicts with keys:
                    {request, model, agent, tool_calls, timestamp, latency_ms, tokens_used}
            model_name: Name of model (e.g. "gpt-5.4", "qwen3-32b.ft-...")
            agent_name: Name of agent (default: tmg_ops_agent)
            timestamp: ISO timestamp (default: now)

        Returns:
            Path to written JSONL file.
        """
        timestamp = timestamp or datetime.utcnow().isoformat()
        filename = f"traces_{agent_name}_{model_name.replace('.', '_')}_{timestamp.replace(':', '-').split('.')[0]}.jsonl"
        filepath = self.traces_dir / filename

        content = "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces)
        self._write_file(filepath, content.encode("utf-8"))

        logger.info(f"Exported {len(traces)} traces to {filepath}")
        return filepath

    def export_eval_results(
        self,
        eval_results: Dict[str, Any],
        eval_run_name: str,
    ) -> Path:
        """
        Export eval run results to Fabric.

        Results dict should include:
          {
            timestamp, eval_pool_size, models: {
              model_name: {ast_accuracy, avg_latency_ms, tokens_used, failures, details}
            }
          }

        Args:
            eval_results: Dict with eval metadata and per-model scores.
            eval_run_name: Name of eval run (e.g. "baseline-vs-student-20260622").

        Returns:
            Path to written JSON file.
        """
        run_dir = self.evals_dir / eval_run_name

        timestamp = datetime.utcnow().isoformat()
        filename = f"eval_results_{timestamp.replace(':', '-').split('.')[0]}.json"
        filepath = run_dir / filename

        content = json.dumps(eval_results, indent=2, ensure_ascii=False)
        self._write_file(filepath, content.encode("utf-8"))

        logger.info(f"Exported eval results to {filepath}")
        return filepath

    def export_eval_details_jsonl(
        self,
        details: List[Dict[str, Any]],
        eval_run_name: str,
    ) -> Path:
        """
        Export per-item eval details as JSONL to Fabric (for Blu to ingest into tables).

        Each record: {item_id, request, reference_call, predicted_calls_by_model, ast_match_by_model, ...}

        Args:
            details: List of per-item eval records.
            eval_run_name: Name of eval run.

        Returns:
            Path to written JSONL file.
        """
        run_dir = self.evals_dir / eval_run_name

        timestamp = datetime.utcnow().isoformat()
        filename = f"eval_details_{timestamp.replace(':', '-').split('.')[0]}.jsonl"
        filepath = run_dir / filename

        content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in details)
        self._write_file(filepath, content.encode("utf-8"))

        logger.info(f"Exported {len(details)} eval detail records to {filepath}")
        return filepath

    def list_export_datasets(self) -> List[str]:
        """
        List available exported datasets in Fabric (for Foundry to consume).

        Returns:
            List of dataset version directories in foundry_exports/.
        """
        if not self.exports_dir.exists():
            return []
        return sorted([d.name for d in self.exports_dir.iterdir() if d.is_dir()])

    def read_export_dataset(self, dataset_version: str) -> List[Dict[str, Any]]:
        """
        Read an exported dataset from Fabric (prepared by Blu for Foundry consumption).

        Expects JSONL file in foundry_exports/<dataset_version>/data.jsonl or *.jsonl

        Args:
            dataset_version: Name of dataset version directory.

        Returns:
            List of records from JSONL.
        """
        dataset_dir = self.exports_dir / dataset_version
        if not dataset_dir.exists():
            logger.warning(f"Dataset not found: {dataset_dir}")
            return []

        records = []
        for jsonl_file in sorted(dataset_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

        logger.info(f"Read {len(records)} records from {dataset_version}")
        return records


def export_tool_eval_to_fabric(
    model_accuracies: Dict[str, float],
    eval_pool_size: int,
    detail_records: List[Dict[str, Any]],
    eval_run_name: str = "baseline-vs-student",
    lakehouse_root: str = "/Volumes/lh_llmops",
    debug: bool = False,
    onelake_workspace: Optional[str] = None,
    onelake_lakehouse: str = "lh_llmops",
) -> Dict[str, Path]:
    """
    Convenience function to export tool-calling eval results to Fabric.

    Args:
        model_accuracies: Dict {model_name: ast_accuracy_pct}
        eval_pool_size: Size of eval pool
        detail_records: List of per-item eval dicts
        eval_run_name: Name of eval run
        lakehouse_root: Fabric Lakehouse root
        debug: Use local debug dir instead

    Returns:
        Dict of {summary_path, details_path}
    """
    exporter = FabricExporter(
        lakehouse_root=lakehouse_root,
        debug=debug,
        onelake_workspace=onelake_workspace,
        onelake_lakehouse=onelake_lakehouse,
    )

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "eval_pool_size": eval_pool_size,
        "eval_run_name": eval_run_name,
        "models": {
            model_name: {
                "ast_accuracy": accuracy,
            }
            for model_name, accuracy in model_accuracies.items()
        },
    }

    summary_path = exporter.export_eval_results(summary, eval_run_name)
    details_path = exporter.export_eval_details_jsonl(detail_records, eval_run_name)

    logger.info(f"Fabric export complete. Summary: {summary_path}, Details: {details_path}")
    return {"summary": summary_path, "details": details_path}
