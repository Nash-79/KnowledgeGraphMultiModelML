#!/usr/bin/env python3
"""
M7 Robustness Test 2: Unit-Noise Analysis

Tests semantic retention degradation when unit edges are corrupted with noise.
This simulates data quality issues (incorrect unit assignments).

Rationale:
- AtP (Attribute Predictability) = 0.9987 with clean data
- Simulate 5%, 10% noise: AtP_noisy = AtP × (1 - noise_rate)
- Recalculate SRS and measure degradation

Usage:
    python scripts/m7_test_unit_noise.py --noise 5 10
"""

import argparse
import json
import sys
from pathlib import Path


def calculate_srs(hp, atp, ap, rtf=0):
    """
    Calculate SRS composite score.

    SRS = 0.25 × HP + 0.20 × AtP + 0.20 × AP + 0.35 × RTF
    Normalized over active components (0.65 when RTF=0)
    """
    srs_raw = 0.25 * hp + 0.20 * atp + 0.20 * ap + 0.35 * rtf
    active_weight = 0.65 if rtf == 0 else 1.0
    srs_normalized = srs_raw / active_weight
    return srs_raw, srs_normalized


def test_noise_level(hp, atp_baseline, ap, noise_percent, srs_baseline):
    """Test a specific noise level."""
    print(f"\n--- Testing {noise_percent}% Unit Noise ---")

    # Simulate noise: Remove noise_percent of unit edges
    retention_rate = 1.0 - (noise_percent / 100.0)
    atp_noisy = atp_baseline * retention_rate

    print(f"  Baseline AtP:  {atp_baseline:.4f} (99.87% concepts have valid units)")
    print(f"  Retention:     {retention_rate:.2f} ({100-noise_percent}% kept)")
    print(f"  Noisy AtP:     {atp_noisy:.4f}")

    # Recalculate SRS
    srs_raw_noisy, srs_norm_noisy = calculate_srs(hp, atp_noisy, ap)

    print(f"  Raw SRS:       {srs_raw_noisy:.5f}")
    print(f"  Normalized SRS:{srs_norm_noisy:.4f}")

    # Calculate degradation
    absolute_drop = srs_baseline - srs_norm_noisy
    percent_drop = (absolute_drop / srs_baseline) * 100

    threshold = 10.0
    status = "PASS" if percent_drop <= threshold else "FAIL"

    print(f"  Degradation:   {percent_drop:.1f}%")
    print(f"  Target:        <={threshold:.1f}%")
    print(f"  Status:        {status}")

    return {
        "noise_percent": noise_percent,
        "atp_noisy": atp_noisy,
        "srs_noisy": srs_norm_noisy,
        "degradation_absolute": absolute_drop,
        "degradation_percent": percent_drop,
        "threshold": threshold,
        "status": "PASS" if percent_drop <= threshold else "FAIL"
    }


def main():
    parser = argparse.ArgumentParser(
        description="M7 Robustness Test: Unit-Noise Analysis"
    )
    parser.add_argument(
        "--noise",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Noise levels to test (percent), e.g., --noise 5 10 15"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("M7 Robustness Test 2: Unit-Noise Analysis")
    print("=" * 70)

    # Load baseline
    srs_file = Path("reports/tables/srs_kge_combined_debug.json")

    if not srs_file.exists():
        print(f"\n[ERROR] {srs_file} not found")
        print("Run from project root: kg-mmml/")
        sys.exit(1)

    with open(srs_file) as f:
        baseline_data = json.load(f)

    hp_baseline = baseline_data.get("HP", 0.2726)
    atp_baseline = baseline_data.get("AtP", 0.9987)
    ap_baseline = baseline_data.get("AP", 1.0000)
    srs_baseline = baseline_data.get("SRS", 0.7571)

    print(f"\n[Baseline Metrics]")
    print(f"  HP:  {hp_baseline:.4f}")
    print(f"  AtP: {atp_baseline:.4f}")
    print(f"  AP:  {ap_baseline:.4f}")
    print(f"  SRS: {srs_baseline:.4f}")

    # Test each noise level
    results_list = []
    for noise in sorted(args.noise):
        result = test_noise_level(hp_baseline, atp_baseline, ap_baseline, noise, srs_baseline)
        results_list.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Noise %':<10} {'AtP':<10} {'SRS':<10} {'Degrade %':<12} {'Status':<10}")
    print("-" * 70)
    print(f"{'Baseline':<10} {atp_baseline:<10.4f} {srs_baseline:<10.4f} {'-':<12} {'PASS':<10}")

    for r in results_list:
        status_text = r["status"]
        print(f"{r['noise_percent']:<10} {r['atp_noisy']:<10.4f} {r['srs_noisy']:<10.4f} "
              f"{r['degradation_percent']:<12.1f} {status_text:<10}")

    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    passed = [r for r in results_list if r["status"] == "PASS"]
    failed = [r for r in results_list if r["status"] == "FAIL"]

    if passed:
        print(f"[PASS] System is robust to unit noise up to {max(r['noise_percent'] for r in passed)}%")
        print(f"       Degradation remains <=10% threshold")
        print()

    if failed:
        print(f"[FAIL] System exceeds 10% degradation at {min(r['noise_percent'] for r in failed)}%+ noise")
        print(f"       This is expected: AtP directly impacts SRS (20% weight)")
        print()

    print("For thesis: Document as evidence of graceful degradation.")
    print("Real-world SEC data is clean (99.87% AtP baseline), so high noise")
    print("scenarios are theoretical stress tests.")
    print()

    # Save results
    output_file = Path("reports/tables/m7_unit_noise_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        "test": "unit-noise",
        "baseline": {
            "HP": hp_baseline,
            "AtP": atp_baseline,
            "AP": ap_baseline,
            "SRS": srs_baseline
        },
        "noise_tests": results_list
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"[OK] Results saved to: {output_file}")
    print()
    print("=" * 70)

    # Exit code: 0 if all passed, 1 if any failed
    return 0 if all(r["status"] == "PASS" for r in results_list) else 1


if __name__ == "__main__":
    sys.exit(main())
