#!/usr/bin/env python3
"""
M8 Master Runner: Execute All Scalability Tests

Runs all M8 scalability tests in sequence and generates consolidated report.

Usage:
    python scripts/run_m8_all.py
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
        description="M8 Master Runner: Execute all scalability tests"
    )
    args = parser.parse_args()

    scripts_dir = Path("scripts")

    if not scripts_dir.exists():
        print("[ERROR] scripts/ directory not found")
        print("Run this from the project root: kg-mmml/")
        sys.exit(1)

    print("=" * 70)
    print("M8 SCALABILITY TESTING SUITE")
    print("Milestone 8: Validating System Scalability")
    print("Week 15-16 Deliverable")
    print("=" * 70)
    print()

    # Test 1: Analytical scaling
    print("[Test 1/3] Analytical Scaling Assessment")
    success_1 = run_script(scripts_dir / "m8_analytical_scale.py")

    # Test 2: Two-hop queries
    print("\n[Test 2/3] Two-Hop Graph Expansion")
    success_2 = run_script(scripts_dir / "m8_test_two_hop.py")

    # Test 3: FAISS parity
    print("\n[Test 3/3] FAISS vs Annoy Parity")
    success_3 = run_script(scripts_dir / "m8_test_faiss_parity.py")

    # Generate consolidated report
    print("\n[Report] Generating Consolidated Results")
    success_4 = run_script(scripts_dir / "m8_generate_report.py")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    status_1 = "COMPLETE" if success_1 else "COMPLETED WITH WARNINGS"
    status_2 = "COMPLETE" if success_2 else "COMPLETED WITH WARNINGS"
    status_3 = "COMPLETE" if success_3 else "COMPLETED WITH WARNINGS"
    status_4 = "COMPLETE" if success_4 else "FAILED"

    print(f"  Scale test:         {status_1}")
    print(f"  Two-hop test:       {status_2}")
    print(f"  FAISS parity test:  {status_3}")
    print(f"  Report generation:  {status_4}")
    print("=" * 70)
    print()

    # Next steps
    print("Results Location:")
    print("   - reports/tables/m8_scalability_results_w15.csv")
    print("   - docs/progress/Week_15-16_M8_Scalability.md")
    print()
    print("Next Steps:")
    print("   1. Review generated reports")
    print("   2. Update consolidated metrics")
    print("   3. Proceed to M9 (Error Analysis + Thesis Writing)")
    print()

    if not all([success_1, success_2, success_3, success_4]):
        print("[WARN] Some tests completed with warnings. Review output above.")
        return 1

    print("[OK] All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
