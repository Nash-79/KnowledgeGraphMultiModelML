#!/usr/bin/env python3
"""
Generate 5 publication-quality figures for Results chapter.

Usage: python scripts/m9_generate_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")


def create_srs_comparison():
    """SRS component comparison before/after RTF."""
    print("Creating Figure 1: SRS Component Comparison...")

    # Data
    components = ['HP', 'AtP', 'AP', 'RTF']
    weights = [0.25, 0.20, 0.20, 0.35]

    # Scores before RTF (RTF=0, renormalized)
    scores_before = [0.2726, 0.9987, 1.0000, 0.0]
    srs_before = 0.7571

    # Scores after RTF
    scores_after = [0.2726, 0.9987, 1.0000, 1.0000]
    srs_after = 0.8179

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Component scores
    x = np.arange(len(components))
    width = 0.35

    bars1 = ax1.bar(x - width/2, scores_before, width, label='Before RTF',
                     color='lightblue', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, scores_after, width, label='After RTF',
                     color='darkblue', edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Score')
    ax1.set_title('SRS Component Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)

    # Right: Overall SRS
    bars = ax2.bar(['Before RTF\n(Renormalized)', 'After RTF\n(Complete)'],
                   [srs_before, srs_after],
                   color=['lightblue', 'darkblue'],
                   edgecolor='black', linewidth=0.5)

    ax2.set_ylabel('SRS Score')
    ax2.set_title('Overall Semantic Retention Score')
    ax2.axhline(y=0.75, color='red', linestyle='--', linewidth=1,
                label='Threshold (0.75)', alpha=0.7)
    ax2.legend()
    ax2.set_ylim([0, 1.0])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n(+{((srs_after-srs_before)/srs_before*100):.1f}%)' if height == srs_after else f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_file = Path("reports/figures/srs_comparison.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def create_latency_scaling():
    """Latency scaling across retrieval methods."""
    print("Creating Figure 2: Latency Scaling...")

    # Data from baseline and M8 analytical projection
    methods = ['Exact\nCosine', 'Filtered\nCosine', 'Annoy', 'FAISS\nHNSW']

    # p99 latencies (ms)
    N1000 = [2.096, 11.538, 0.032, 0.175]
    N3218 = [5.483, 2.429, 0.037, 0.255]
    N10000 = [17.039, 2.429, 0.042, 0.291]  # Projected

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, N1000, width, label='N=1,000',
                   color='lightgreen', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, N3218, width, label='N=3,218',
                   color='green', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, N10000, width, label='N=10,000 (projected)',
                   color='darkgreen', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('p99 Latency (ms)')
    ax.set_title('Retrieval Latency Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.axhline(y=150, color='red', linestyle='--', linewidth=1,
               label='SLO (150ms)', alpha=0.7)

    # Use log scale for better visibility
    ax.set_yscale('log')
    ax.set_ylim([0.01, 200])

    plt.tight_layout()

    output_file = Path("reports/figures/latency_scaling.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def create_f1_distribution():
    """F1 score distribution across concepts."""
    print("Creating Figure 3: F1 Score Distribution...")

    # Load error analysis data
    analysis_file = Path("reports/tables/m9_error_analysis_detailed.csv")
    df = pd.read_csv(analysis_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Histogram of F1 scores
    ax1.hist(df['f1_score'], bins=20, color='steelblue',
             edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('F1 Score')
    ax1.set_ylabel('Number of Concepts')
    ax1.set_title('Distribution of Per-Concept F1 Scores')
    ax1.axvline(x=0.99, color='red', linestyle='--', linewidth=1,
                label='F1=0.99', alpha=0.7)
    ax1.legend()

    # Right: F1 by support size
    support_bins = pd.cut(df['support'], bins=[0, 250, 500, 750, 1000],
                          labels=['<250', '250-500', '500-750', '>750'])

    support_data = []
    labels = []
    for bin_label in ['<250', '250-500', '500-750', '>750']:
        mask = support_bins == bin_label
        if mask.any():
            support_data.append(df[mask]['f1_score'].values)
            labels.append(f'{bin_label}\n(n={mask.sum()})')

    bp = ax2.boxplot(support_data, labels=labels, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')

    ax2.set_xlabel('Support (Training Examples)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Support Size')
    ax2.set_ylim([0.97, 1.01])
    ax2.axhline(y=0.99, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    output_file = Path("reports/figures/f1_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def create_robustness_degradation():
    """Robustness under perturbation."""
    print("Creating Figure 4: Robustness Degradation...")

    # Data from M7 robustness tests
    tests = ['Baseline', 'Taxonomy\nRemoval', 'Unit Noise\n5%', 'Unit Noise\n10%']
    srs_scores = [0.7571, 0.6150, 0.7045, 0.6891]
    degradation = [0.0, -18.8, -7.0, -9.0]
    colors = ['green', 'red', 'orange', 'orange']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: SRS scores
    bars = ax1.bar(tests, srs_scores, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.7)

    ax1.set_ylabel('SRS Score')
    ax1.set_title('SRS Under Perturbation')
    ax1.axhline(y=0.75, color='blue', linestyle='--', linewidth=1,
                label='Threshold (0.75)', alpha=0.7)
    ax1.legend()
    ax1.set_ylim([0, 0.9])

    # Add value labels
    for bar, score in zip(bars, srs_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., score,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=9)

    # Right: Degradation percentage
    bars = ax2.bar(tests[1:], degradation[1:],
                   color=['red', 'orange', 'orange'],
                   edgecolor='black', linewidth=0.5, alpha=0.7)

    ax2.set_ylabel('SRS Degradation (%)')
    ax2.set_title('Degradation vs Baseline')
    ax2.axhline(y=-10, color='blue', linestyle='--', linewidth=1,
                label='Target (≤10%)', alpha=0.7)
    ax2.legend()
    ax2.set_ylim([-25, 0])

    # Add value labels
    for bar, deg in zip(bars, degradation[1:]):
        ax2.text(bar.get_x() + bar.get_width()/2., deg,
                f'{deg:.1f}%',
                ha='center', va='top', fontsize=9)

    plt.tight_layout()

    output_file = Path("reports/figures/robustness_degradation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def create_performance_by_category():
    """Performance by financial statement category."""
    print("Creating Figure 5: Performance by Category...")

    # Load error analysis by category
    category_file = Path("reports/tables/m9_error_by_category.csv")
    df = pd.read_csv(category_file, index_col=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by F1 score
    df_sorted = df.sort_values('f1_score', ascending=True)

    # Create horizontal bar chart
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['f1_score'],
                   color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted.index)
    ax.set_xlabel('Average F1 Score')
    ax.set_title('Classification Performance by Financial Statement Category')
    ax.set_xlim([0.97, 1.01])
    ax.axvline(x=0.99, color='red', linestyle='--', linewidth=1,
               label='F1=0.99', alpha=0.7)
    ax.legend()

    # Add value labels with support count
    for i, (bar, (cat, row)) in enumerate(zip(bars, df_sorted.iterrows())):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {width:.4f} (n={int(row["support"])})',
                ha='left', va='center', fontsize=9)

    plt.tight_layout()

    output_file = Path("reports/figures/performance_by_category.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def main():
    print("\n" + "=" * 70)
    print("M9 Figure Generation: Creating Thesis Visualizations")
    print("=" * 70 + "\n")

    # Create all figures
    create_srs_comparison()
    create_latency_scaling()
    create_f1_distribution()
    create_robustness_degradation()
    create_performance_by_category()

    print("\n" + "=" * 70)
    print("COMPLETE: All figures generated")
    print("=" * 70)

    print("\nFigures saved to reports/figures/:")
    print("  1. srs_comparison.png - SRS before/after RTF")
    print("  2. latency_scaling.png - Latency across scales")
    print("  3. f1_distribution.png - F1 score distribution")
    print("  4. robustness_degradation.png - Robustness tests")
    print("  5. performance_by_category.png - Performance by category")
    print("\nAll figures are publication-quality (300 DPI)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
