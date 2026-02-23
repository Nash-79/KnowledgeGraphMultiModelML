#!/usr/bin/env python3
"""
M10 master runner: multi-seed statistical validation + markdown report generation.
"""

import json
import pathlib
import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"FAILED: {description} (exit={result.returncode})")
        sys.exit(result.returncode)
    print(f"COMPLETED: {description}")
    return result


def _format_summary_table(summary_df):
    try:
        return summary_df.to_markdown(index=False)
    except Exception:
        return "```text\n" + summary_df.to_string(index=False) + "\n```"


def generate_markdown_report():
    import pandas as pd

    tests_path = pathlib.Path("reports/tables/m10_statistical_tests.json")
    summary_path = pathlib.Path("reports/tables/m10_statistical_summary.csv")
    if not tests_path.exists() or not summary_path.exists():
        print("WARNING: M10 statistical outputs not found; report generation skipped.")
        return None

    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    summary_df = pd.read_csv(summary_path)

    report = f"""# M10: Statistical Validation Report

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Milestone**: M10 (Week 19-20)
**Seeds**: {", ".join(map(str, tests["seeds"]))}

---

## Executive Summary

- Micro-F1 improvement: {tests["micro_f1"]["improvement_pp_mean"]:+.2f}pp
  95% CI: [{tests["micro_f1"]["improvement_pp_95ci"][0]:+.2f}, {tests["micro_f1"]["improvement_pp_95ci"][1]:+.2f}]
  p-value: {tests["micro_f1"]["paired_t_test"]["p_value"]:.6f}
- Macro-F1 improvement: {tests["macro_f1"]["improvement_pp_mean"]:+.2f}pp
  95% CI: [{tests["macro_f1"]["improvement_pp_95ci"][0]:+.2f}, {tests["macro_f1"]["improvement_pp_95ci"][1]:+.2f}]
  p-value: {tests["macro_f1"]["paired_t_test"]["p_value"]:.6f}

## Statistical Summary

{_format_summary_table(summary_df)}

## Decision Gate Note

- Target micro-F1 gain: +3.0pp
- Observed mean micro-F1 gain: {tests["micro_f1"]["improvement_pp_mean"]:+.2f}pp
- Interpretation: ceiling-effect context should be retained in thesis narrative.

## Thesis Integration

Update:
1. `docs/Architecture.md` with confidence intervals and p-values where relevant.
2. `../README.md` headline metrics if any canonical values change.
3. `docs/DOCUMENT_INDEX.md` if canonical ownership changes.
"""

    out_path = pathlib.Path("docs/M10_STATISTICAL_VALIDATION_REPORT.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Generated report: {out_path}")
    return out_path


def main():
    start_time = datetime.now()
    print("=" * 70)
    print("M10 STATISTICAL VALIDATION")
    print("=" * 70)

    run_command(
        [sys.executable, "scripts/m10_statistical_validation.py"],
        "M10 statistical validation (5 seeds + CI + t-tests)",
    )

    print("\n" + "=" * 70)
    print("GENERATING: M10 markdown report")
    print("=" * 70)
    generate_markdown_report()

    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("M10 COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration:.1f}s")
    print("Outputs:")
    print(" - reports/tables/m10_seed{42..46}_baseline_text_metrics.json")
    print(" - reports/tables/m10_seed{42..46}_text_concept_metrics.json")
    print(" - reports/tables/m10_statistical_summary.csv")
    print(" - reports/tables/m10_statistical_tests.json")
    print(" - docs/M10_STATISTICAL_VALIDATION_REPORT.md")


if __name__ == "__main__":
    main()
