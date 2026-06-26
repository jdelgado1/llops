#!/usr/bin/env python
"""
Export production (or test) traces to Fabric Lakehouse.

For live deployment: Instrument Foundry Hosted Agent to capture traces.
For testing: Generate mock traces to verify Fabric pipeline.

Usage:
  # Test with mock traces
  python scripts/export_traces_to_fabric.py --mock --fabric-debug
  
  # Real production (post-deployment with instrumentation)
  python scripts/export_traces_to_fabric.py \
    --traces-jsonl artifacts/production_traces.jsonl \
    --fabric-lakehouse-root /Volumes/lh_llmops \
    --fabric-export
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from llmops.fabric_integration import FabricExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_traces(num_traces: int = 10) -> List[Dict[str, Any]]:
    """
    Generate mock production traces for testing Fabric pipeline.
    
    In real deployment, these would come from Foundry Tracing API.
    """
    mock_traces = []
    
    requests = [
        "Open a high priority support ticket for customer CU-9012 about a billing error.",
        "Check data usage for account ACC-5021 in the current billing cycle.",
        "Reset the modem for customer CU-8834 and schedule a technician visit for tomorrow.",
        "Search the web for current outages in the southeast region.",
        "Get network status for broadband services in the northeast region.",
    ]
    
    for i in range(num_traces):
        trace = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": "qwen3-32b.ft-example",  # Will be the trained student model
            "agent": "tmg_ops_agent",
            "request": requests[i % len(requests)],
            "predicted_tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "create_support_ticket" if i % 2 == 0 else "get_network_status",
                        "arguments": json.dumps({
                            "customer_id": f"CU-{9000 + i}",
                            "priority": "high"
                        })
                    }
                }
            ],
            "latency_ms": 120 + (i % 50),
            "tokens_used": 80 + (i % 30),
        }
        mock_traces.append(trace)
    
    logger.info(f"Generated {len(mock_traces)} mock traces")
    return mock_traces


def load_traces_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load traces from JSONL file."""
    traces = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    logger.info(f"Loaded {len(traces)} traces from {jsonl_path}")
    return traces


def main():
    parser = argparse.ArgumentParser(
        description="Export Foundry production traces to Fabric Lakehouse."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate mock traces for testing (no real deployment yet).",
    )
    parser.add_argument(
        "--traces-jsonl",
        default=None,
        help="Path to real traces JSONL (from Foundry Tracing API export).",
    )
    parser.add_argument(
        "--model-name",
        default="qwen3-32b.ft-distilled",
        help="Model name/version for traces.",
    )
    parser.add_argument(
        "--agent-name",
        default="tmg_ops_agent",
        help="Agent name for traces.",
    )
    parser.add_argument(
        "--fabric-lakehouse-root",
        default="/Volumes/lh_llmops",
        help="Fabric Lakehouse root path.",
    )
    parser.add_argument(
        "--fabric-export",
        action="store_true",
        help="Export to Fabric (requires write access).",
    )
    parser.add_argument(
        "--fabric-debug",
        action="store_true",
        help="Use local ./fabric_export_debug/ instead of Fabric.",
    )
    parser.add_argument(
        "--num-mock-traces",
        type=int,
        default=10,
        help="Number of mock traces to generate (if --mock).",
    )

    args = parser.parse_args()

    # Load or generate traces
    if args.mock:
        traces = generate_mock_traces(num_traces=args.num_mock_traces)
    elif args.traces_jsonl:
        traces = load_traces_from_jsonl(args.traces_jsonl)
    else:
        logger.error("Must provide --mock or --traces-jsonl")
        return

    # Export to Fabric
    exporter = FabricExporter(
        lakehouse_root=args.fabric_lakehouse_root,
        debug=args.fabric_debug,
    )

    if args.fabric_export or args.fabric_debug:
        filepath = exporter.export_traces(
            traces=traces,
            model_name=args.model_name,
            agent_name=args.agent_name,
        )
        logger.info(f"✅ Exported {len(traces)} traces to: {filepath}")
        logger.info(f"   Blu can ingest from: Files/llmops/raw/foundry_traces/")
    else:
        logger.info("(Use --fabric-export or --fabric-debug to write to Fabric)")


if __name__ == "__main__":
    main()
