#!/usr/bin/env python3
"""
M10: Statistical Validation with Multiple Random Seeds

Runs baseline and text+concept experiments with 5 random seeds (42, 43, 44, 45, 46)
to validate reproducibility and compute confidence intervals.

This addresses the academic rigour requirement for MSc thesis by demonstrating
that results are statistically significant and not artifacts of a single train/test split.

Usage:
    # Run all experiments (baseline text-only + text+concept) for 5 seeds
    python scripts/m10_statistical_validation.py --run_experiments

    # Compute statistics from saved results
    python scripts/m10_statistical_validation.py --compute_statistics

    # Run both (default)
    python scripts/m10_statistical_validation.py

Outputs:
    - reports/tables/m10_seed{X}_baseline_text_metrics.json (X=42,43,44,45,46)
    - reports/tables/m10_seed{X}_text_concept_metrics.json
    - reports/tables/m10_statistical_summary.csv (mean, std, 95% CI)
    - reports/tables/m10_statistical_tests.json (paired t-test results)
"""
import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


# Configuration
SEEDS = [42, 43, 44, 45, 46]
FACTS_PATH = "data/processed/sec_edgar/facts.jsonl"
TAXONOMY_PATH = "datasets/sec_edgar/taxonomy/usgaap_combined.csv"
CONCEPT_FEATURES_NPZ = "data/processed/sec_edgar/features/concept_features_filing.npz"
CONCEPT_FEATURES_INDEX = "data/processed/sec_edgar/features/concept_features_index.csv"
OUTPUT_DIR = pathlib.Path("reports/tables")
TEST_SIZE = 0.25  # 25% test split (matches baseline_tfidf.py default)


def run_baseline_experiment(seed: int) -> Dict:
    """
    Run text-only baseline experiment for a given seed.

    Returns:
        metrics dict with micro_f1, macro_f1, etc.
    """
    output_path = OUTPUT_DIR / f"m10_seed{seed}_baseline_text_metrics.json"

    cmd = [
        sys.executable, "-m", "src.cli.baseline_tfidf",
        "--facts", FACTS_PATH,
        "--taxonomy", TAXONOMY_PATH,
        "--out", str(output_path),
        "--random_state", str(seed),
        "--test_size", str(TEST_SIZE),
    ]

    print(f"\n{'='*60}")
    print(f"Running baseline (text-only) with seed={seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Baseline experiment failed for seed {seed}")
        print(result.stderr)
        raise RuntimeError(f"Baseline experiment failed for seed {seed}")

    print(result.stdout)

    # Load and return metrics
    metrics = json.loads(output_path.read_text())
    print(f"OK Baseline seed={seed}: micro_f1={metrics['micro_f1']:.4f}, macro_f1={metrics['macro_f1']:.4f}")

    return metrics


def run_text_concept_experiment(seed: int) -> Dict:
    """
    Run text+concept experiment for a given seed.

    Returns:
        metrics dict with micro_f1, macro_f1, etc.
    """
    output_path = OUTPUT_DIR / f"m10_seed{seed}_text_concept_metrics.json"

    cmd = [
        sys.executable, "-m", "src.cli.baseline_tfidf",
        "--facts", FACTS_PATH,
        "--taxonomy", TAXONOMY_PATH,
        "--concept_features_npz", CONCEPT_FEATURES_NPZ,
        "--concept_features_index", CONCEPT_FEATURES_INDEX,
        "--out", str(output_path),
        "--random_state", str(seed),
        "--test_size", str(TEST_SIZE),
    ]

    print(f"\n{'='*60}")
    print(f"Running text+concept with seed={seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Text+concept experiment failed for seed {seed}")
        print(result.stderr)
        raise RuntimeError(f"Text+concept experiment failed for seed {seed}")

    print(result.stdout)

    # Load and return metrics
    metrics = json.loads(output_path.read_text())
    print(f"OK Text+concept seed={seed}: micro_f1={metrics['micro_f1']:.4f}, macro_f1={metrics['macro_f1']:.4f}")

    return metrics


def run_all_experiments():
    """Run all experiments for all seeds."""
    print("\n" + "="*60)
    print("M10 STATISTICAL VALIDATION: Running experiments")
    print("="*60)
    print(f"Seeds: {SEEDS}")
    print(f"Test size: {TEST_SIZE} ({int(TEST_SIZE*100)}% of data)")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_results = []
    text_concept_results = []

    for seed in SEEDS:
        # Run baseline (text-only)
        baseline_metrics = run_baseline_experiment(seed)
        baseline_results.append(baseline_metrics)

        # Run text+concept
        text_concept_metrics = run_text_concept_experiment(seed)
        text_concept_results.append(text_concept_metrics)

    print("\n" + "="*60)
    print("OK All experiments completed successfully")
    print("="*60)

    return baseline_results, text_concept_results


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for a list of values.

    Args:
        values: List of metric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (mean, std, ci_lower, ci_upper)
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std (n-1 denominator)
    se = std / np.sqrt(n)  # Standard error

    # t-distribution critical value for 95% CI
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_critical * se

    ci_lower = mean - margin
    ci_upper = mean + margin

    return mean, std, ci_lower, ci_upper


def paired_t_test(baseline_values: List[float], treatment_values: List[float]) -> Dict:
    """
    Perform paired t-test comparing baseline vs treatment.

    Returns:
        dict with t_statistic, p_value, significant (p<0.05)
    """
    t_stat, p_value = stats.ttest_rel(treatment_values, baseline_values)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01),
    }


def compute_statistics():
    """
    Load saved experiment results and compute statistical summary.
    """
    print("\n" + "="*60)
    print("M10 STATISTICAL VALIDATION: Computing statistics")
    print("="*60)

    # Load all results
    baseline_micro_f1 = []
    baseline_macro_f1 = []
    text_concept_micro_f1 = []
    text_concept_macro_f1 = []

    for seed in SEEDS:
        # Load baseline
        baseline_path = OUTPUT_DIR / f"m10_seed{seed}_baseline_text_metrics.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline results for seed {seed}: {baseline_path}")

        baseline = json.loads(baseline_path.read_text())
        baseline_micro_f1.append(baseline["micro_f1"])
        baseline_macro_f1.append(baseline["macro_f1"])

        # Load text+concept
        text_concept_path = OUTPUT_DIR / f"m10_seed{seed}_text_concept_metrics.json"
        if not text_concept_path.exists():
            raise FileNotFoundError(f"Missing text+concept results for seed {seed}: {text_concept_path}")

        text_concept = json.loads(text_concept_path.read_text())
        text_concept_micro_f1.append(text_concept["micro_f1"])
        text_concept_macro_f1.append(text_concept["macro_f1"])

    print(f"OK Loaded results for {len(SEEDS)} seeds")

    # Compute statistics for baseline
    bl_micro_mean, bl_micro_std, bl_micro_ci_low, bl_micro_ci_high = compute_confidence_interval(baseline_micro_f1)
    bl_macro_mean, bl_macro_std, bl_macro_ci_low, bl_macro_ci_high = compute_confidence_interval(baseline_macro_f1)

    # Compute statistics for text+concept
    tc_micro_mean, tc_micro_std, tc_micro_ci_low, tc_micro_ci_high = compute_confidence_interval(text_concept_micro_f1)
    tc_macro_mean, tc_macro_std, tc_macro_ci_low, tc_macro_ci_high = compute_confidence_interval(text_concept_macro_f1)

    # Compute improvements (in percentage points)
    micro_improvements = [(tc - bl) * 100 for tc, bl in zip(text_concept_micro_f1, baseline_micro_f1)]
    macro_improvements = [(tc - bl) * 100 for tc, bl in zip(text_concept_macro_f1, baseline_macro_f1)]

    micro_imp_mean, micro_imp_std, micro_imp_ci_low, micro_imp_ci_high = compute_confidence_interval(micro_improvements)
    macro_imp_mean, macro_imp_std, macro_imp_ci_low, macro_imp_ci_high = compute_confidence_interval(macro_improvements)

    # Statistical significance tests
    micro_f1_test = paired_t_test(baseline_micro_f1, text_concept_micro_f1)
    macro_f1_test = paired_t_test(baseline_macro_f1, text_concept_macro_f1)

    # Create summary dataframe
    summary = pd.DataFrame([
        {
            "model": "Baseline (text-only)",
            "micro_f1_mean": f"{bl_micro_mean:.4f}",
            "micro_f1_std": f"{bl_micro_std:.4f}",
            "micro_f1_95ci": f"[{bl_micro_ci_low:.4f}, {bl_micro_ci_high:.4f}]",
            "macro_f1_mean": f"{bl_macro_mean:.4f}",
            "macro_f1_std": f"{bl_macro_std:.4f}",
            "macro_f1_95ci": f"[{bl_macro_ci_low:.4f}, {bl_macro_ci_high:.4f}]",
        },
        {
            "model": "Text+Concept (KG-as-features)",
            "micro_f1_mean": f"{tc_micro_mean:.4f}",
            "micro_f1_std": f"{tc_micro_std:.4f}",
            "micro_f1_95ci": f"[{tc_micro_ci_low:.4f}, {tc_micro_ci_high:.4f}]",
            "macro_f1_mean": f"{tc_macro_mean:.4f}",
            "macro_f1_std": f"{tc_macro_std:.4f}",
            "macro_f1_95ci": f"[{tc_macro_ci_low:.4f}, {tc_macro_ci_high:.4f}]",
        },
        {
            "model": "Improvement (pp)",
            "micro_f1_mean": f"{micro_imp_mean:+.2f}",
            "micro_f1_std": f"{micro_imp_std:.3f}",
            "micro_f1_95ci": f"[{micro_imp_ci_low:+.2f}, {micro_imp_ci_high:+.2f}]",
            "macro_f1_mean": f"{macro_imp_mean:+.2f}",
            "macro_f1_std": f"{macro_imp_std:.3f}",
            "macro_f1_95ci": f"[{macro_imp_ci_low:+.2f}, {macro_imp_ci_high:+.2f}]",
        },
    ])

    # Save summary
    summary_path = OUTPUT_DIR / "m10_statistical_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nOK Saved statistical summary: {summary_path}")
    print("\n" + summary.to_string(index=False))

    # Save statistical tests
    tests = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "confidence_level": 0.95,
        "micro_f1": {
            "baseline_mean": bl_micro_mean,
            "text_concept_mean": tc_micro_mean,
            "improvement_pp_mean": micro_imp_mean,
            "improvement_pp_95ci": [micro_imp_ci_low, micro_imp_ci_high],
            "paired_t_test": micro_f1_test,
        },
        "macro_f1": {
            "baseline_mean": bl_macro_mean,
            "text_concept_mean": tc_macro_mean,
            "improvement_pp_mean": macro_imp_mean,
            "improvement_pp_95ci": [macro_imp_ci_low, macro_imp_ci_high],
            "paired_t_test": macro_f1_test,
        },
    }

    tests_path = OUTPUT_DIR / "m10_statistical_tests.json"
    tests_path.write_text(json.dumps(tests, indent=2))
    print(f"\nOK Saved statistical tests: {tests_path}")

    # Print statistical test results
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test)")
    print("="*60)
    print(f"\nMicro-F1:")
    print(f"  Improvement: {micro_imp_mean:+.2f}pp [95% CI: {micro_imp_ci_low:+.2f}, {micro_imp_ci_high:+.2f}]")
    print(f"  t-statistic: {micro_f1_test['t_statistic']:.4f}")
    print(f"  p-value: {micro_f1_test['p_value']:.4f}")
    print(f"  Significant (p<0.05): {micro_f1_test['significant_p05']}")
    print(f"  Significant (p<0.01): {micro_f1_test['significant_p01']}")

    print(f"\nMacro-F1:")
    print(f"  Improvement: {macro_imp_mean:+.2f}pp [95% CI: {macro_imp_ci_low:+.2f}, {macro_imp_ci_high:+.2f}]")
    print(f"  t-statistic: {macro_f1_test['t_statistic']:.4f}")
    print(f"  p-value: {macro_f1_test['p_value']:.4f}")
    print(f"  Significant (p<0.05): {macro_f1_test['significant_p05']}")
    print(f"  Significant (p<0.01): {macro_f1_test['significant_p01']}")

    print("\n" + "="*60)
    print("OK Statistical validation complete")
    print("="*60)

    return summary, tests


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_experiments", action="store_true",
                    help="Run all experiments for 5 seeds")
    ap.add_argument("--compute_statistics", action="store_true",
                    help="Compute statistics from saved results")
    args = ap.parse_args()

    # Default: run both if no flags specified
    if not args.run_experiments and not args.compute_statistics:
        args.run_experiments = True
        args.compute_statistics = True

    if args.run_experiments:
        run_all_experiments()

    if args.compute_statistics:
        compute_statistics()


if __name__ == "__main__":
    main()
