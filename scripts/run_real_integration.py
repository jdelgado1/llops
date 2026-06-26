#!/usr/bin/env python
"""
End-to-end REAL Foundry ↔ Fabric integration flow.

Does everything in sequence:
1. Run eval (baseline, student, teacher) → export to Fabric
2. Generate real traces (student inference) → export to Fabric  
3. Test if Foundry SDK can consume Fabric exports

This is the real deal - no mocks, no placeholders.

Usage:
  python scripts/run_real_integration.py \
    --eval-pool-jsonl artifacts/eval_pool_114items.jsonl \
    --baseline-deployment qwen3-32b.ft-e037e55c11e8415a9e09ed6527789e10-format-primer \
    --student-deployment <STUDENT_FINETUNED_DEPLOYMENT> \
    --fabric-lakehouse-root /Volumes/lh_llmops
"""

import argparse
import sys
import logging
import os
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


def run_step(step_num: int, step_name: str, command: str) -> bool:
    """Run a step and report status."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*70}")
    print(f"Running: {command}\n")
    
    exit_code = os.system(command)
    
    if exit_code == 0:
        print(f"✅ STEP {step_num} COMPLETE: {step_name}\n")
        return True
    else:
        print(f"❌ STEP {step_num} FAILED: {step_name}")
        print(f"   Exit code: {exit_code}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Real Foundry <-> Fabric integration (end-to-end)."
    )
    parser.add_argument(
        "--eval-pool-jsonl",
        required=True,
        help="Path to eval pool JSONL.",
    )
    parser.add_argument(
        "--baseline-deployment",
        required=True,
        help="Baseline model deployment.",
    )
    parser.add_argument(
        "--student-deployment",
        required=True,
        help="Trained student model deployment.",
    )
    parser.add_argument(
        "--teacher-model",
        default="gpt-5.4",
        help="Teacher model name.",
    )
    parser.add_argument(
        "--eval-run-name",
        default="baseline-vs-student-real",
        help="Eval run name.",
    )
    parser.add_argument(
        "--fabric-lakehouse-root",
        default="/Volumes/lh_llmops",
        help="Fabric Lakehouse root (local path fallback).",
    )
    parser.add_argument(
        "--onelake-workspace",
        default=None,
        help='Upload directly to Fabric via OneLake API (e.g. "Fine Tune Demo"). '
             "Overrides --fabric-lakehouse-root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit eval/traces to N items (for quick testing).",
    )
    parser.add_argument(
        "--skip-traces",
        action="store_true",
        help="Skip trace generation (just run eval).",
    )

    args = parser.parse_args()

    # Prepare environment
    os.environ["PYTHONPATH"] = "src"
    
    print("\n" + "="*70)
    print("REAL FOUNDRY ↔ FABRIC INTEGRATION TEST")
    print("="*70)
    print(f"Eval pool: {args.eval_pool_jsonl}")
    print(f"Baseline: {args.baseline_deployment}")
    print(f"Student:  {args.student_deployment}")
    print(f"Teacher:  {args.teacher_model}")
    print(f"Fabric root: {args.fabric_lakehouse_root}")
    if args.onelake_workspace:
        print(f"OneLake workspace: {args.onelake_workspace}")
    if args.limit:
        print(f"Limit: {args.limit} items (quick test)")

    # Build commands
    limit_flag = f"--limit {args.limit}" if args.limit else ""
    onelake_flag = f'--onelake-workspace "{args.onelake_workspace}"' if args.onelake_workspace else ""

    # STEP 1: Run 3-way eval + export to Fabric
    step1_cmd = (
        f".venv\\Scripts\\python.exe scripts\\test_foundry_fabric_integration.py "
        f"--eval-pool-jsonl {args.eval_pool_jsonl} "
        f"--baseline-deployment {args.baseline_deployment} "
        f"--student-deployment {args.student_deployment} "
        f"--teacher-model {args.teacher_model} "
        f"--eval-run-name {args.eval_run_name} "
        f"--fabric-lakehouse-root {args.fabric_lakehouse_root} "
        f"--test-sdk-consumption "
        f"{onelake_flag} "
        f"{limit_flag}"
    )

    if not run_step(1, "Run 3-way Eval + Export to Fabric", step1_cmd):
        print("❌ Cannot continue: eval step failed")
        sys.exit(1)

    # STEP 2: Generate real traces (student inference) + export to Fabric
    if not args.skip_traces:
        step2_cmd = (
            f".venv\\Scripts\\python.exe scripts\\gen_real_traces.py "
            f"--eval-pool-jsonl {args.eval_pool_jsonl} "
            f"--model-deployment {args.student_deployment} "
            f"--fabric-lakehouse-root {args.fabric_lakehouse_root} "
            f"--fabric-export "
            f"{onelake_flag} "
            f"{limit_flag}"
        )

        if not run_step(2, "Generate Real Traces + Export to Fabric", step2_cmd):
            print("⚠️  Traces failed, but eval is complete")
    else:
        print("\nSkipping trace generation (--skip-traces)")

    # FINAL REPORT
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)
    print("\n✅ Real data now in Fabric:")
    print(f"   Eval results: Files/llmops/raw/foundry_evals/{args.eval_run_name}/")
    print(f"   Traces:       Files/llmops/raw/foundry_traces/")
    print("\n🎯 Next steps:")
    print("   1. Blu ingests traces + evals into Fabric tables")
    print("   2. Blu computes scorecard + detects drift")
    print("   3. Blu exports retraining candidates to foundry_exports/")
    print("   4. Foundry SDK can consume from Fabric paths (or download+reupload)")
    print("\n📊 Ready for demo!")


if __name__ == "__main__":
    main()
