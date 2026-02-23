#!/usr/bin/env python3
"""
SRS Stability Verification Script

Since SRS metrics (HP, AtP, AP) are deterministic structural measures computed
from the knowledge graph topology, this script verifies that:
1. Multiple runs produce identical results (no randomness)
2. The computation is reproducible across different executions

For future work with embedding-based RTF (Relational Type Fidelity), this script
can be extended to test stability across different random seeds for embedding
initialization.

Usage:
    python scripts/compute_srs_stability.py \
        --config configs/experiment_kge_enhanced.yaml \
        --runs 5 \
        --output reports/tables/srs_stability_w9.csv
"""
import argparse
import csv
import json
import os
import pathlib
import sys
import numpy as np
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from cli.compute_srs import (
    find_snapshot_folder,
    load_nodes_edges,
    metric_atp,
    metric_hp_coverage,
    metric_ap_directionality,
    weighted_srs,
)


def load_config(config_path):
    """Load YAML config file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_srs_once(kg_folder, srs_weights):
    """Run SRS computation once and return metrics dict."""
    concepts, units, periods, edges_by_type, all_edges = load_nodes_edges(kg_folder)
    
    # Compute structural proxies
    atp = metric_atp(concepts, edges_by_type)
    hp = metric_hp_coverage(concepts, edges_by_type)
    apdir = metric_ap_directionality(edges_by_type)
    rtf = None  # RTF requires embeddings; not implemented yet
    
    scores = {"RTF": rtf, "AP": apdir, "HP": hp, "AtP": atp}
    srs = weighted_srs(scores, srs_weights)
    
    return {
        "RTF": rtf,
        "AP": apdir,
        "HP": hp,
        "AtP": atp,
        "SRS": srs,
        "n_concepts": len(concepts),
        "n_units": len(units),
        "n_periods": len(periods),
        "n_edges_isa": len(edges_by_type.get("is-a", [])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--runs", type=int, default=5, help="Number of stability runs (default: 5)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    srs_weights = cfg.get("eval", {}).get("srs", {}).get("weights", {})
    kg_snapshot = cfg.get("data", {}).get("kg_snapshot", None)
    kg_folder = find_snapshot_folder(kg_snapshot)
    
    print(f"[SRS Stability] Configuration:")
    print(f"  Config: {args.config}")
    print(f"  KG Snapshot: {kg_folder}")
    print(f"  SRS Weights: {srs_weights}")
    print(f"  Runs: {args.runs}")
    print()
    
    # Run SRS computation multiple times
    results = []
    for run_id in range(args.runs):
        metrics = compute_srs_once(kg_folder, srs_weights)
        results.append(metrics)
        print(f"[Run {run_id+1}/{args.runs}] HP={metrics['HP']:.6f}, AtP={metrics['AtP']:.6f}, AP={metrics['AP']:.6f}, SRS={metrics['SRS']:.6f}")
    
    # Compute statistics (mean, std, min, max)
    metric_names = ["HP", "AtP", "AP", "SRS"]
    stats = {}
    
    for metric in metric_names:
        values = [r[metric] for r in results]
        stats[metric] = {
            "mean": np.mean(values),
            "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
            "min": np.min(values),
            "max": np.max(values),
            "values": values,
        }
    
    # Check for perfect stability (std = 0)
    all_stable = all(stats[m]["std"] == 0.0 for m in metric_names)
    
    print()
    print("="*80)
    print("STABILITY ANALYSIS")
    print("="*80)
    print()
    print(f"{'Metric':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Status':<15}")
    print("-"*80)
    
    for metric in metric_names:
        mean = stats[metric]["mean"]
        std = stats[metric]["std"]
        min_val = stats[metric]["min"]
        max_val = stats[metric]["max"]
        status = "✅ STABLE" if std == 0.0 else f"⚠️  VAR (σ={std:.6f})"
        print(f"{metric:<10} {mean:.6f}     {std:.6f}     {min_val:.6f}     {max_val:.6f}     {status}")
    
    print("-"*80)
    print()
    
    if all_stable:
        print("✅ ALL METRICS PERFECTLY STABLE (σ=0.000 across all runs)")
        print("   → SRS computation is deterministic (as expected for structural metrics)")
        print("   → No randomness in topology-based HP, AtP, AP calculations")
    else:
        print("⚠️  VARIANCE DETECTED")
        print("   → This is unexpected for deterministic structural metrics")
        print("   → Investigate potential sources of non-determinism")
    
    # Decision gate validation
    srs_mean = stats["SRS"]["mean"]
    srs_std = stats["SRS"]["std"]
    gate_threshold = 0.05  # Stability threshold: std < 0.05
    
    print()
    print("="*80)
    print(f"DECISION GATE: SRS stability (std < {gate_threshold})")
    print("="*80)
    print(f"  SRS Mean: {srs_mean:.6f}")
    print(f"  SRS Std:  {srs_std:.6f}")
    print(f"  Threshold: {gate_threshold}")
    print(f"  Status: {'✅ PASS' if srs_std < gate_threshold else '❌ FAIL'}")
    print("="*80)
    print()
    
    # Save results to CSV
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["metric", "mean", "std", "min", "max", "ci_lower_95", "ci_upper_95", "runs_tested"])
        
        # Data rows
        for metric in metric_names:
            mean = stats[metric]["mean"]
            std = stats[metric]["std"]
            min_val = stats[metric]["min"]
            max_val = stats[metric]["max"]
            
            # 95% confidence interval (±1.96σ for normal distribution)
            # For deterministic metrics (σ=0), CI collapses to point estimate
            ci_lower = mean - 1.96 * std
            ci_upper = mean + 1.96 * std
            
            writer.writerow([
                metric,
                f"{mean:.6f}",
                f"{std:.6f}",
                f"{min_val:.6f}",
                f"{max_val:.6f}",
                f"{ci_lower:.6f}",
                f"{ci_upper:.6f}",
                args.runs,
            ])
        
        # Decision gate row
        gate_status = "PASS" if srs_std < gate_threshold else "FAIL"
        writer.writerow([
            "DECISION_GATE",
            gate_status,
            f"std_threshold={gate_threshold}",
            f"srs_std={srs_std:.6f}",
            "",
            "",
            "",
            "",
        ])
    
    print(f"Output written to: {output_path}")
    
    # Also save detailed JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "kg_snapshot": kg_folder,
            "runs": args.runs,
            "stats": {m: {k: (v if k != "values" else v) for k, v in s.items()} for m, s in stats.items()},
            "decision_gate": {
                "threshold": gate_threshold,
                "srs_std": srs_std,
                "status": gate_status,
            },
            "metadata": results[0],  # First run metadata (n_concepts, etc.)
        }, f, indent=2)
    
    print(f"Detailed JSON written to: {json_path}")
    print()
    
    # Return exit code based on decision gate
    return 0 if srs_std < gate_threshold else 1


if __name__ == "__main__":
    exit(main())
