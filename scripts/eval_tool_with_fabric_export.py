#!/usr/bin/env python
"""
Tool-calling eval runner: 3-way comparison (baseline vs student vs teacher).
Exports results to Fabric Lakehouse for Blu's pipeline.

Usage:
  python scripts/eval_tool_with_fabric_export.py \
    --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
    --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
    --student-deployment <STUDENT_FINETUNED_DEPLOYMENT from Foundry> \
    --eval-run-name baseline-vs-student-20260622 \
    --fabric-lakehouse-root /Volumes/lh_llmops \
    --fabric-export
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from llmops.tooldata import ToolCallItem
from llmops.ast_check import ast_accuracy_for_item
from llmops.models import invoke_model
from llmops.fabric_integration import export_tool_eval_to_fabric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_pool(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load eval pool JSONL (ToolCallItem dicts)."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded eval pool: {len(records)} items from {jsonl_path}")
    return records


def run_model_on_pool(
    model_name: str,
    deployment: str,
    eval_pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run a model on eval pool and compute AST accuracy.

    Returns:
        {
          model_name, deployment, ast_accuracy, num_correct, num_total,
          failures (list of error records), latency_ms_avg, tokens_used_total
        }
    """
    logger.info(f"Running {model_name} ({deployment}) on {len(eval_pool)} items...")

    correct_count = 0
    total = len(eval_pool)
    failures = []
    total_latency_ms = 0
    total_tokens = 0

    for i, item in enumerate(eval_pool):
        try:
            # Invoke model
            result = invoke_model(
                deployment=deployment,
                messages=item.get("messages", []),
                tools=item.get("tools", []),
                model_name=model_name,
            )

            predicted_calls = result.get("tool_calls", [])
            reference_calls = item.get("reference_tool_calls", [])

            # Compute AST accuracy
            is_correct = ast_accuracy_for_item(
                predicted_calls=predicted_calls,
                reference_calls=reference_calls,
            )

            if is_correct:
                correct_count += 1

            total_latency_ms += result.get("latency_ms", 0)
            total_tokens += result.get("tokens_used", 0)

            if (i + 1) % 20 == 0:
                logger.info(f"  {i + 1}/{total} completed...")

        except Exception as e:
            logger.warning(f"Error on item {i}: {e}")
            failures.append(
                {
                    "item_idx": i,
                    "error": str(e),
                }
            )

    accuracy = (correct_count / total * 100) if total > 0 else 0
    avg_latency = total_latency_ms / total if total > 0 else 0

    logger.info(
        f"{model_name} AST accuracy: {accuracy:.2f}% ({correct_count}/{total}) "
        f"| Latency avg: {avg_latency:.1f}ms | Tokens: {total_tokens}"
    )

    return {
        "model_name": model_name,
        "deployment": deployment,
        "ast_accuracy": accuracy,
        "num_correct": correct_count,
        "num_total": total,
        "failures": failures,
        "latency_ms_avg": avg_latency,
        "tokens_used_total": total_tokens,
    }


def build_detail_records(
    eval_pool: List[Dict[str, Any]],
    model_results: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Build per-item eval detail records for Fabric ingestion.

    Each record includes item_id, reference_call, predictions from all models, AST match per model.
    """
    details = []
    for i, item in enumerate(eval_pool):
        detail = {
            "eval_item_id": i,
            "request": item.get("messages", [{}])[-1].get("content", "")[:500],  # user msg
            "reference_tool_calls": item.get("reference_tool_calls", []),
            "predicted_by_model": {},
            "ast_match_by_model": {},
        }

        for model_name, predictions in model_results.items():
            if i < len(predictions):
                detail["predicted_by_model"][model_name] = predictions[i].get(
                    "tool_calls", []
                )
                detail["ast_match_by_model"][model_name] = predictions[i].get(
                    "is_correct", False
                )

        details.append(detail)

    return details


def main():
    parser = argparse.ArgumentParser(
        description="Run 3-way tool-calling eval and export to Fabric."
    )
    parser.add_argument(
        "--eval-pool-jsonl",
        required=True,
        help="Path to eval pool JSONL (ToolCallItem format).",
    )
    parser.add_argument(
        "--baseline-deployment",
        required=True,
        help="Baseline model deployment name.",
    )
    parser.add_argument(
        "--student-deployment",
        required=True,
        help="Distilled student model deployment name.",
    )
    parser.add_argument(
        "--teacher-model",
        default="gpt-5.4",
        help="Teacher model name (default: gpt-5.4).",
    )
    parser.add_argument(
        "--eval-run-name",
        default="baseline-vs-student",
        help="Name of eval run for Fabric export.",
    )
    parser.add_argument(
        "--fabric-lakehouse-root",
        default="/Volumes/lh_llmops",
        help="Fabric Lakehouse root path.",
    )
    parser.add_argument(
        "--fabric-export",
        action="store_true",
        help="Export results to Fabric (requires write access to lakehouse).",
    )
    parser.add_argument(
        "--fabric-debug",
        action="store_true",
        help="Use local ./fabric_export_debug/ instead of Fabric paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit eval pool to N items (for testing).",
    )

    args = parser.parse_args()

    # Load eval pool
    eval_pool = load_eval_pool(args.eval_pool_jsonl)
    if args.limit:
        eval_pool = eval_pool[: args.limit]
        logger.info(f"Limited to {args.limit} items for testing.")

    # Run 3-way eval
    models_to_eval = [
        ("baseline", args.baseline_deployment),
        ("student", args.student_deployment),
        ("teacher", args.teacher_model),
    ]

    results = {}
    model_accuracies = {}

    for model_label, deployment in models_to_eval:
        result = run_model_on_pool(model_label, deployment, eval_pool)
        results[model_label] = result
        model_accuracies[model_label] = result["ast_accuracy"]

    # Print summary
    logger.info("\n=== EVAL SUMMARY ===")
    for model_label, result in results.items():
        logger.info(
            f"{model_label:12s}: {result['ast_accuracy']:6.2f}% "
            f"({result['num_correct']}/{result['num_total']}) "
        )

    # Export to Fabric
    if args.fabric_export:
        logger.info(f"\nExporting results to Fabric ({args.eval_run_name})...")

        detail_records = build_detail_records(eval_pool, results)

        export_paths = export_tool_eval_to_fabric(
            model_accuracies=model_accuracies,
            eval_pool_size=len(eval_pool),
            detail_records=detail_records,
            eval_run_name=args.eval_run_name,
            lakehouse_root=args.fabric_lakehouse_root,
            debug=args.fabric_debug,
        )

        logger.info(f"✅ Fabric export complete:")
        logger.info(f"   Summary: {export_paths['summary']}")
        logger.info(f"   Details: {export_paths['details']}")
        logger.info(f"   Blu can now ingest from: Files/llmops/raw/foundry_evals/{args.eval_run_name}/")
    else:
        logger.info("\n(No Fabric export; use --fabric-export to enable)")


if __name__ == "__main__":
    main()
