#!/usr/bin/env python
"""
Generate REAL traces by running inference on the trained student model.

Output: production traces (model calls) → export to Fabric.

Usage:
  python scripts/gen_real_traces.py \
    --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
    --model-deployment <STUDENT_FINETUNED_DEPLOYMENT> \
    --output-traces artifacts/real_traces_student.jsonl \
    --fabric-lakehouse-root /Volumes/lh_llmops \
    --fabric-export
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from llmops.models import invoke_model
from llmops.ast_check import check_ast
from llmops.fabric_integration import FabricExporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_eval_pool(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load eval pool JSONL."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded eval pool: {len(records)} items from {jsonl_path}")
    return records


def generate_traces(
    model_deployment: str,
    eval_pool: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run model inference on eval pool → capture as production traces.
    
    Each trace is one model call (request + tool_calls).
    """
    logger.info(f"\n🔄 Generating traces by running {model_deployment}...")
    
    traces = []
    
    for i, item in enumerate(eval_pool):
        try:
            # Run inference
            result = invoke_model(
                deployment=model_deployment,
                messages=item.get("messages", []),
                tools=item.get("tools", []),
                model_name=model_deployment,
            )
            
            # Capture as trace
            trace = {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_deployment,
                "agent": "tmg_ops_agent",
                "request": item.get("messages", [{}])[-1].get("content", "")[:500],
                "predicted_tool_calls": result.get("tool_calls", []),
                "latency_ms": result.get("latency_ms", 0),
                "tokens_used": result.get("tokens_used", 0),
            }
            traces.append(trace)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{len(eval_pool)} traces...")
                
        except Exception as e:
            logger.warning(f"Trace {i} failed: {e}")
            continue
    
    logger.info(f"✅ Generated {len(traces)} real traces")
    return traces


def save_traces(traces: List[Dict[str, Any]], output_path: str) -> Path:
    """Save traces to JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    
    logger.info(f"✅ Saved traces to {output_path}")
    return output_path


def export_traces_to_fabric(
    traces: List[Dict[str, Any]],
    model_deployment: str,
    fabric_lakehouse_root: str,
    onelake_workspace: str = None,
) -> Path:
    """Export traces to Fabric."""
    exporter = FabricExporter(
        lakehouse_root=fabric_lakehouse_root,
        debug=False,
        onelake_workspace=onelake_workspace,
    )
    
    export_path = exporter.export_traces(
        traces=traces,
        model_name=model_deployment,
        agent_name="tmg_ops_agent",
    )
    
    logger.info(f"✅ Exported {len(traces)} traces to Fabric")
    logger.info(f"   Path: Files/llmops/raw/foundry_traces/")
    logger.info(f"   Blu can ingest from: {export_path}")
    
    return export_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate real traces by running inference on trained model."
    )
    parser.add_argument(
        "--eval-pool-jsonl",
        required=True,
        help="Path to eval pool JSONL.",
    )
    parser.add_argument(
        "--model-deployment",
        required=True,
        help="Trained student model deployment name.",
    )
    parser.add_argument(
        "--output-traces",
        default="artifacts/real_traces_student.jsonl",
        help="Output file for traces.",
    )
    parser.add_argument(
        "--fabric-lakehouse-root",
        default="/Volumes/lh_llmops",
        help="Fabric Lakehouse root path (local path fallback).",
    )
    parser.add_argument(
        "--onelake-workspace",
        default=None,
        help='Upload directly to Fabric via OneLake API (e.g. "Fine Tune Demo"). '
             "Overrides --fabric-lakehouse-root.",
    )
    parser.add_argument(
        "--fabric-export",
        action="store_true",
        help="Also export to Fabric.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N items (for testing).",
    )

    args = parser.parse_args()

    # Load eval pool
    eval_pool = load_eval_pool(args.eval_pool_jsonl)
    
    if args.limit:
        eval_pool = eval_pool[:args.limit]
        logger.info(f"Limited to {args.limit} items")

    # Generate traces
    traces = generate_traces(args.model_deployment, eval_pool)
    
    # Save locally
    local_path = save_traces(traces, args.output_traces)
    
    # Export to Fabric
    if args.fabric_export:
        fabric_path = export_traces_to_fabric(
            traces, args.model_deployment, args.fabric_lakehouse_root,
            onelake_workspace=args.onelake_workspace,
        )
        logger.info(f"\n✅ Traces ready:")
        logger.info(f"   Local: {local_path}")
        logger.info(f"   Fabric: {fabric_path}")
    else:
        logger.info(f"\n✅ Traces saved locally: {local_path}")
        logger.info(f"   Use --fabric-export to push to Fabric")


if __name__ == "__main__":
    main()
