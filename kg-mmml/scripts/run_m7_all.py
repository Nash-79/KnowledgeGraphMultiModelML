#!/usr/bin/env python3
"""
M7 Master Runner: Execute All Robustness Tests

Runs all M7 robustness tests in sequence and generates consolidated report.

Usage:
    python scripts/run_m7_all.py

    # Or with custom noise levels:
    python scripts/run_m7_all.py --noise 5 10 15 20
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name, args=None):
    """Run a Python script and capture output."""
    cmd = [sys.executable, str(script_name)]
    if args:
        cmd.extend(args)

    print(f"\n{'=' * 70}")
    print(f"Running: {script_name.name}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"\n[WARN] {script_name.name} returned non-zero exit code")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="M7 Master Runner: Execute all robustness tests"
    )
    parser.add_argument(
        "--noise",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Noise levels to test (percent), default: 5 10"
    )
    args = parser.parse_args()

    scripts_dir = Path("scripts")

    if not scripts_dir.exists():
        print("[ERROR] scripts/ directory not found")
        print("Run this from the project root: kg-mmml/")
        sys.exit(1)

    print("=" * 70)
    print("M7 ROBUSTNESS TESTING SUITE")
    print("Milestone 7: Validating System Robustness")
    print("Week 13-14 Deliverable")
    print("=" * 70)
    print()

    # Test 1: Taxonomy-off
    print("[Test 1/2] Taxonomy Removal Analysis")
    success_1 = run_script(scripts_dir / "m7_test_taxonomy_off.py")

    # Test 2: Unit-noise
    print("\n[Test 2/2] Unit-Edge Corruption Analysis")
    noise_args = ["--noise"] + [str(n) for n in args.noise]
    success_2 = run_script(scripts_dir / "m7_test_unit_noise.py", noise_args)

    # Generate consolidated report
    print("\n[Report] Generating Consolidated Results")
    success_3 = run_script(scripts_dir / "m7_generate_report.py")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    status_1 = "COMPLETE" if success_1 else "COMPLETED WITH WARNINGS"
    status_2 = "COMPLETE" if success_2 else "COMPLETED WITH WARNINGS"
    status_3 = "COMPLETE" if success_3 else "FAILED"

    print(f"  Taxonomy-off test:     {status_1}")
    print(f"  Unit-noise test:       {status_2}")
    print(f"  Report generation:     {status_3}")
    print("=" * 70)
    print()

    # Next steps
    print("Results Location:")
    print("   - reports/tables/m7_robustness_results_w13.csv")
    print("   - docs/progress/Week_13-14_M7_Robustness.md")
    print()
    print("Next Steps:")
    print("   1. Review generated reports")
    print("   2. Update consolidated metrics if needed")
    print("   3. Proceed to M8 (Scalability) or M9 (Thesis Writing)")
    print()

    if not all([success_1, success_2, success_3]):
        print("[WARN] Some tests completed with warnings. Review output above.")
        return 1

    print("[OK] All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
