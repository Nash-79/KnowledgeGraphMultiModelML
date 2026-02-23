#!/usr/bin/env python3
"""
Compare baseline (text-only) vs joint model (text+concept features).
Validates the +3pp micro-F1 improvement decision gate.

Usage:
    python scripts/compare_baseline_vs_joint.py \
        --baseline reports/tables/baseline_text_seed42_metrics.json \
        --joint outputs/joint_no_penalty/metrics.json \
        --output reports/tables/baseline_vs_joint_comparison.csv
"""
import argparse
import json
import pathlib
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="Path to baseline metrics JSON")
    ap.add_argument("--joint", required=True, help="Path to joint model metrics JSON")
    ap.add_argument("--output", required=True, help="Path to output comparison CSV")
    ap.add_argument("--gate_threshold", type=float, default=3.0, 
                    help="Micro-F1 improvement threshold in percentage points (default: 3.0)")
    args = ap.parse_args()
    
    # Load metrics
    baseline_path = pathlib.Path(args.baseline)
    joint_path = pathlib.Path(args.joint)
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_path}")
    if not joint_path.exists():
        raise FileNotFoundError(f"Joint metrics not found: {joint_path}")
    
    baseline = json.loads(baseline_path.read_text())
    joint = json.loads(joint_path.read_text())
    
    # Extract test metrics
    baseline_macro = baseline.get("macro_f1", 0.0)
    baseline_micro = baseline.get("micro_f1", 0.0)
    
    # Joint model stores test metrics under "test" key
    if "test" in joint:
        joint_macro = joint["test"].get("macro_f1", 0.0)
        joint_micro = joint["test"].get("micro_f1", 0.0)
    else:
        joint_macro = joint.get("macro_f1", 0.0)
        joint_micro = joint.get("micro_f1", 0.0)
    
    # Compute improvements (in percentage points)
    macro_improvement = (joint_macro - baseline_macro) * 100
    micro_improvement = (joint_micro - baseline_micro) * 100
    
    # Decision gate validation
    gate_passed = micro_improvement >= args.gate_threshold
    
    # Create comparison dataframe
    comparison = pd.DataFrame([
        {
            "model": "text-only (baseline)",
            "macro_f1": f"{baseline_macro:.4f}",
            "micro_f1": f"{baseline_micro:.4f}",
            "n_train": baseline.get("n_docs_train", "N/A"),
            "n_test": baseline.get("n_docs_test", "N/A"),
        },
        {
            "model": "text+concept (joint, λ=0.0)",
            "macro_f1": f"{joint_macro:.4f}",
            "micro_f1": f"{joint_micro:.4f}",
            "n_train": joint.get("n_train", "N/A"),
            "n_test": joint.get("n_test", "N/A"),
        },
        {
            "model": "IMPROVEMENT (pp)",
            "macro_f1": f"{macro_improvement:+.2f}",
            "micro_f1": f"{micro_improvement:+.2f}",
            "n_train": "",
            "n_test": "",
        },
        {
            "model": f"DECISION GATE (≥{args.gate_threshold}pp micro-F1)",
            "macro_f1": "",
            "micro_f1": "✅ PASS" if gate_passed else "❌ FAIL",
            "n_train": "",
            "n_test": "",
        }
    ])
    
    # Save CSV
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE vs JOINT MODEL COMPARISON")
    print("="*60)
    print(f"\n{comparison.to_string(index=False)}\n")
    print("="*60)
    print(f"Output written to: {output_path}")
    print("="*60)
    
    if gate_passed:
        print("✅ Decision gate PASSED: Joint model improves micro-F1 by ≥3pp")
    else:
        print(f"❌ Decision gate FAILED: Joint model improves micro-F1 by only {micro_improvement:.2f}pp (threshold: {args.gate_threshold}pp)")
    
    return 0 if gate_passed else 1


if __name__ == "__main__":
    exit(main())
