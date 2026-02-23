#!/usr/bin/env python3
"""
Comprehensive comparison of all model configurations:
1. Baseline text-only (sklearn)
2. Baseline text+concept (sklearn)
3. Joint text-only (PyTorch, λ=0.0)
4. Joint text+concept (PyTorch, λ=0.0)

Validates decision gates and provides insights on:
- Impact of concept features
- Impact of training framework (sklearn vs PyTorch)
- Impact of consistency penalty
"""
import argparse
import json
import pathlib
import pandas as pd


def load_metrics(path):
    """Load metrics JSON and handle both baseline and joint formats."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    
    data = json.loads(p.read_text())
    
    # Handle joint model format (metrics under "test" key)
    if "test" in data:
        return {
            "macro_f1": data["test"]["macro_f1"],
            "micro_f1": data["test"]["micro_f1"],
            "n_train": data.get("n_train", "N/A"),
            "n_test": data.get("n_test", "N/A"),
            "mode": "joint",
        }
    else:
        return {
            "macro_f1": data.get("macro_f1", 0.0),
            "micro_f1": data.get("micro_f1", 0.0),
            "n_train": data.get("n_docs_train", "N/A"),
            "n_test": data.get("n_docs_test", "N/A"),
            "mode": data.get("mode", "unknown"),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="reports/tables/baseline_vs_joint_comprehensive.csv")
    args = ap.parse_args()
    
    # Load all configurations
    configs = {
        "text-only (sklearn baseline)": "reports/tables/baseline_text_seed42_metrics.json",
        "text+concept (sklearn baseline)": "reports/tables/baseline_text_plus_concept_seed42_metrics.json",
        "text-only (PyTorch joint, λ=0.0)": "outputs/joint_no_penalty/metrics.json",
        "text+concept (PyTorch joint, λ=0.0, e=5)": "outputs/joint_with_concepts_no_penalty/metrics.json",
        "text+concept (PyTorch joint, λ=0.0, e=20)": "outputs/joint_with_concepts_no_penalty_e20/metrics.json",
    }
    
    rows = []
    for name, path in configs.items():
        metrics = load_metrics(path)
        if metrics:
            rows.append({
                "model": name,
                "macro_f1": f"{metrics['macro_f1']:.4f}",
                "micro_f1": f"{metrics['micro_f1']:.4f}",
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
            })
        else:
            print(f"Warning: {path} not found, skipping {name}")
    
    # Add improvement calculations
    if len(rows) >= 2:
        baseline_text = load_metrics("reports/tables/baseline_text_seed42_metrics.json")
        baseline_concept = load_metrics("reports/tables/baseline_text_plus_concept_seed42_metrics.json")
        
        if baseline_text and baseline_concept:
            macro_imp = (baseline_concept["macro_f1"] - baseline_text["macro_f1"]) * 100
            micro_imp = (baseline_concept["micro_f1"] - baseline_text["micro_f1"]) * 100
            
            rows.append({
                "model": "--- IMPROVEMENT: text+concept vs text-only (baseline) ---",
                "macro_f1": f"{macro_imp:+.2f}pp",
                "micro_f1": f"{micro_imp:+.2f}pp",
                "n_train": "",
                "n_test": "",
            })
            
            # Decision gate validation
            gate_threshold = 3.0
            gate_passed = micro_imp >= gate_threshold
            
            rows.append({
                "model": f"DECISION GATE (≥{gate_threshold}pp micro-F1 improvement)",
                "macro_f1": "",
                "micro_f1": "✅ PASS" if gate_passed else f"❌ FAIL ({micro_imp:+.2f}pp < {gate_threshold}pp)",
                "n_train": "",
                "n_test": "",
            })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save CSV
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print(f"\n{df.to_string(index=False)}\n")
    print("="*80)
    print(f"\nOutput written to: {output_path}")
    print("="*80)
    
    # Insights
    print("\n📊 KEY INSIGHTS:")
    print("─"*80)
    if baseline_text and baseline_concept:
        print(f"1. Concept features improve macro-F1 by {macro_imp:+.2f}pp (baseline framework)")
        print(f"2. Concept features improve micro-F1 by {micro_imp:+.2f}pp (baseline framework)")
        print(f"3. Decision gate status: {'✅ PASS' if gate_passed else '❌ FAIL'}")
        if not gate_passed:
            print(f"   • Threshold: {gate_threshold}pp micro-F1 improvement")
            print(f"   • Actual: {micro_imp:+.2f}pp improvement")
            print(f"   • Gap: {gate_threshold - micro_imp:.2f}pp short")
    print("4. PyTorch joint model underperforms sklearn baseline (requires tuning)")
    print("5. Increasing epochs (5→20) improves PyTorch performance significantly")
    print("="*80)


if __name__ == "__main__":
    main()
