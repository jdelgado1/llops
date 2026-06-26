#!/usr/bin/env python
"""
Quickly regenerate eval_details JSONL with per-item student correctness.
Uses cached requests where possible to avoid re-evaluating all 64 items.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

from llmops.fabric_integration import FabricExporter
from llmops.tooldata import load_toolcalling
from llmops.ast_check import check_ast
from llmops.models import invoke_model
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_eval_pool(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load eval pool JSONL."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"✅ Loaded eval pool: {len(records)} items")
    return records


def run_eval_on_model(
    model_name: str,
    deployment: str,
    eval_pool: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> tuple[int, List[bool]]:
    """Run eval on a single model, return (correct_count, per_item_results)."""
    if limit:
        eval_pool = eval_pool[:limit]
    
    logger.info(f"\n🔄 Running {model_name} ({deployment}) on {len(eval_pool)} items...")
    
    correct_count = 0
    per_item_results = []
    
    for i, item in enumerate(eval_pool):
        is_correct = False
        try:
            result = invoke_model(
                deployment=deployment,
                messages=item.get("messages", []),
                tools=item.get("tools", []),
                model_name=model_name,
            )
            
            predicted_calls = result.get("tool_calls", [])
            reference_calls = item.get("reference_tool_calls", [])
            tools = item.get("tools", [])
            
            ast_result = check_ast(
                pred_calls=predicted_calls,
                ref_calls=reference_calls,
                tools=tools,
            )
            is_correct = ast_result.correct
            
            if is_correct:
                correct_count += 1
            
            # Log progress every 5 items
            if (i + 1) % 5 == 0:
                logger.info(f"  Progress: {i + 1}/{len(eval_pool)} items completed")
                
        except Exception as e:
            logger.warning(f"  Item {i}: {e}")
        
        per_item_results.append(is_correct)
    
    accuracy = (correct_count / len(eval_pool) * 100) if eval_pool else 0
    logger.info(f"✅ {model_name}: {accuracy:.1f}% AST accuracy ({correct_count}/{len(eval_pool)})")
    
    return correct_count, per_item_results


def upload_to_fabric(
    eval_details: List[Dict[str, Any]],
    eval_results_json: Dict[str, Any],
    onelake_workspace: str,
    eval_run_name: str,
) -> None:
    """Upload eval_details and eval_results to Fabric."""
    logger.info(f"\n🔄 Uploading to Fabric ({onelake_workspace})...")
    
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    exporter = FabricExporter(
        lakehouse_root="/Volumes/lh_llmops",
        debug=False,
        onelake_workspace=onelake_workspace,
        onelake_lakehouse="lh_llmops",
    )
    
    # Export eval_details JSONL
    details_filename = f"eval_details_{timestamp}.jsonl"
    details_path = f"llmops/raw/foundry_evals/{eval_run_name}/{details_filename}"
    
    details_content = "\n".join(json.dumps(record) for record in eval_details)
    exporter._write_file(details_path, details_content)
    logger.info(f"✅ Uploaded eval_details: {details_path}")
    
    # Export eval_results JSON
    results_filename = f"eval_results_{timestamp}.json"
    results_path = f"llmops/raw/foundry_evals/{eval_run_name}/{results_filename}"
    
    results_content = json.dumps(eval_results_json, indent=2)
    exporter._write_file(results_path, results_content)
    logger.info(f"✅ Uploaded eval_results: {results_path}")
    
    # Print verification
    logger.info(f"\n✅ Fabric upload complete!")
    logger.info(f"   eval_details ({len(eval_details)} items):")
    for model_label, accuracies in _compute_model_stats(eval_details).items():
        pct = 100.0 * accuracies["true"] / len(eval_details)
        logger.info(f"      {model_label}: {accuracies['true']} true, {accuracies['false']} false ({pct:.1f}%)")


def _compute_model_stats(eval_details: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Compute per-model true/false counts from eval_details."""
    models = set()
    for detail in eval_details:
        models.update(detail.get("ast_match_by_model", {}).keys())
    
    stats = {}
    for model in sorted(models):
        true_count = sum(
            1 for detail in eval_details
            if detail.get("ast_match_by_model", {}).get(model, False)
        )
        false_count = len(eval_details) - true_count
        stats[model] = {"true": true_count, "false": false_count}
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate eval_details with per-item student correctness."
    )
    parser.add_argument("--eval-pool-jsonl", required=True, help="Path to eval pool JSONL.")
    parser.add_argument("--baseline-deployment", required=True, help="Baseline model deployment.")
    parser.add_argument("--student-deployment", required=True, help="Student model deployment.")
    parser.add_argument("--teacher-model", default="gpt-5.4", help="Teacher model name.")
    parser.add_argument("--onelake-workspace", required=True, help="Fabric workspace name (e.g., 'Fine Tune Demo').")
    parser.add_argument("--eval-run-name", default="baseline-vs-student-corrected", help="Eval run name.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N items for testing.")

    args = parser.parse_args()

    # Load eval pool
    eval_pool = load_eval_pool(args.eval_pool_jsonl)
    eval_pool_size = len(eval_pool)
    
    if args.limit:
        eval_pool = eval_pool[:args.limit]
        logger.info(f"Limited to {args.limit} items for testing")

    # Run 3-way eval with per-item tracking
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Running 3-way eval (per-item tracking)")
    logger.info("="*60)
    
    models_to_eval = [
        ("baseline", args.baseline_deployment),
        ("student", args.student_deployment),
        ("teacher", args.teacher_model),
    ]
    
    eval_results_dict = {}
    for model_label, deployment in models_to_eval:
        try:
            correct_count, per_item = run_eval_on_model(model_label, deployment, eval_pool, args.limit)
            eval_results_dict[model_label] = {
                "correct": correct_count,
                "total": len(eval_pool),
                "per_item": per_item,
            }
        except Exception as e:
            logger.error(f"❌ Eval failed for {model_label}: {e}")
            sys.exit(1)

    # Build eval_details with per-item correctness
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Building eval_details JSONL")
    logger.info("="*60)
    
    eval_details = []
    for i in range(len(eval_pool)):
        detail = {
            "eval_item_id": i,
            "ast_match_by_model": {
                model_label: results["per_item"][i]
                for model_label, results in eval_results_dict.items()
            }
        }
        eval_details.append(detail)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVAL SUMMARY")
    logger.info("="*60)
    for model_label, results in eval_results_dict.items():
        pct = 100.0 * results["correct"] / results["total"]
        logger.info(f"{model_label:12s}: {pct:6.2f}% ({results['correct']}/{results['total']})")
    
    # Build eval_results JSON
    eval_results_json = {
        "timestamp": datetime.now().isoformat(),
        "eval_pool_size": len(eval_pool),
        "eval_run_name": args.eval_run_name,
        "models": {
            model_label: 100.0 * results["correct"] / results["total"]
            for model_label, results in eval_results_dict.items()
        }
    }
    
    # Upload to Fabric
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Uploading to Fabric")
    logger.info("="*60)
    
    try:
        upload_to_fabric(
            eval_details=eval_details,
            eval_results_json=eval_results_json,
            onelake_workspace=args.onelake_workspace,
            eval_run_name=args.eval_run_name,
        )
    except Exception as e:
        logger.error(f"❌ Fabric upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
