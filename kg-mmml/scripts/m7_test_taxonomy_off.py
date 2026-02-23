#!/usr/bin/env python3
"""
M7 Robustness Test 1: Taxonomy-Off Analysis

Tests semantic retention degradation when taxonomy is removed.
Analytical test based on existing SRS component metrics.

Usage: python scripts/m7_test_taxonomy_off.py
"""

import json
import sys
from pathlib import Path


def load_baseline_srs():
    """Load baseline SRS metrics from Phase B results."""
    srs_file = Path("reports/tables/srs_kge_combined_debug.json")

    if not srs_file.exists():
        print(f"Error: {srs_file} not found")
        print("Run from project root: kg-mmml/")
        sys.exit(1)

    with open(srs_file) as f:
        data = json.load(f)

    return data


def calculate_srs(hp, atp, ap, rtf=0):
    """
    Calculate SRS composite score.
    SRS = 0.25 × HP + 0.20 × AtP + 0.20 × AP + 0.35 × RTF
    Since RTF is not implemented, renormalize over active components (0.65).
    """
    srs_raw = 0.25 * hp + 0.20 * atp + 0.20 * ap + 0.35 * rtf
    active_weight = 0.65 if rtf == 0 else 1.0
    srs_normalized = srs_raw / active_weight
    return srs_raw, srs_normalized


def main():
    print("\nM7 Robustness Test 1: Taxonomy-Off Analysis\n")

    # Load baseline metrics
    print("Loading baseline SRS metrics...")
    baseline_data = load_baseline_srs()

    hp_baseline = baseline_data.get("HP", 0.2726)
    atp_baseline = baseline_data.get("AtP", 0.9987)
    ap_baseline = baseline_data.get("AP", 1.0000)
    srs_baseline = baseline_data.get("SRS", 0.7571)

    print(f"  Baseline HP:  {hp_baseline:.4f}")
    print(f"  Baseline AtP: {atp_baseline:.4f}")
    print(f"  Baseline AP:  {ap_baseline:.4f}")
    print(f"  Baseline SRS: {srs_baseline:.4f}\n")

    # Verify baseline calculation
    print("Verifying baseline SRS calculation...")
    srs_raw_baseline, srs_norm_baseline = calculate_srs(hp_baseline, atp_baseline, ap_baseline)

    print(f"  Raw SRS:        {srs_raw_baseline:.5f}")
    print(f"  Normalized SRS: {srs_norm_baseline:.4f}")
    print(f"  Expected SRS:   {srs_baseline:.4f}")

    if abs(srs_norm_baseline - srs_baseline) > 0.001:
        print("  Warning: Calculated SRS differs from baseline")
    else:
        print("  Calculation verified\n")

    # Test: Taxonomy off (HP = 0)
    print("Simulating taxonomy removal (HP set to 0)...")
    hp_no_taxonomy = 0.0
    srs_raw_no_tax, srs_norm_no_tax = calculate_srs(hp_no_taxonomy, atp_baseline, ap_baseline)

    print(f"  HP without taxonomy: {hp_no_taxonomy:.4f}")
    print(f"  Raw SRS:             {srs_raw_no_tax:.5f}")
    print(f"  Normalized SRS:      {srs_norm_no_tax:.4f}\n")

    # Calculate degradation
    print("Measuring degradation...")
    absolute_drop = srs_baseline - srs_norm_no_tax
    percent_drop = (absolute_drop / srs_baseline) * 100

    threshold = 10.0
    status = "PASS" if percent_drop <= threshold else "FAIL"

    print(f"  Baseline SRS:     {srs_baseline:.4f}")
    print(f"  Taxonomy-off SRS: {srs_norm_no_tax:.4f}")
    print(f"  Absolute drop:    {absolute_drop:.4f}")
    print(f"  Percent drop:     {percent_drop:.1f}%")
    print(f"  Target:           <={threshold:.1f}%")
    print(f"  Status:           {status}\n")

    # Interpretation
    print("Interpretation:")
    if percent_drop > threshold:
        print(f"SRS degradation of {percent_drop:.1f}% exceeds the {threshold}% threshold.")
        print("This demonstrates that taxonomy contributes meaningfully to semantic")
        print("preservation and that the auto-generated hierarchy provides measurable")
        print("value. Document as 'controlled degradation' in thesis.\n")
    else:
        print(f"SRS degradation of {percent_drop:.1f}% passes the threshold, suggesting")
        print("concept features capture semantic value independently of hierarchy.\n")

    # Save results
    output_file = Path("reports/tables/m7_taxonomy_off_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "test": "taxonomy-off",
        "baseline": {
            "HP": hp_baseline,
            "AtP": atp_baseline,
            "AP": ap_baseline,
            "SRS": srs_baseline
        },
        "perturbed": {
            "HP": hp_no_taxonomy,
            "AtP": atp_baseline,
            "AP": ap_baseline,
            "SRS": srs_norm_no_tax
        },
        "degradation": {
            "absolute": absolute_drop,
            "percent": percent_drop,
            "threshold": threshold,
            "status": "PASS" if percent_drop <= threshold else "FAIL"
        }
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}\n")

    return 0 if percent_drop <= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
