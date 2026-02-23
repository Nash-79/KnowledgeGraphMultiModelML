#!/usr/bin/env python3
"""
SRS Metrics Comparison Visualization (W5-6 vs W7-8)

Generates bar chart showing progression of AtP, HP, AP, and SRS metrics.

Usage:
    python scripts/visualization/plot_srs_comparison.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Ensure output directory exists
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Data from PR3 (W5-6) and PR4 (W7-8)
metrics = ['AtP', 'HP', 'AP', 'SRS']
w5_6 = [0.998, 0.0115, 1.000, 0.670]
w7_8 = [0.9987, 0.2726, 1.000, 0.7571]
gates = [0.95, 0.25, 0.99, 0.75]

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 6))

# Bar positions
x = np.arange(len(metrics))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, w5_6, width, label='Week 5-6 (PR3)', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, w7_8, width, label='Week 7-8 (PR4)', 
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, gates, width, label='Decision Gate', 
               color='#e74c3c', alpha=0.5, edgecolor='black', linewidth=1.2, hatch='//')

# Add value labels on bars
def autolabel(bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}' if val < 1 else f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(bars1, w5_6)
autolabel(bars2, w7_8)
autolabel(bars3, gates)

# Styling
ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('SRS Metrics Progression: Week 5-6 → Week 7-8\n' + 
             'HP Improvement: +2370% (Auto-Taxonomy Impact)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Add annotations for key improvements
ax.annotate('HP: +237%', xy=(1, 0.2726), xytext=(1.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.annotate('SRS: +13%', xy=(3, 0.7571), xytext=(3.5, 0.9),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# Add gate status indicators
for i, (metric, w7_val, gate_val) in enumerate(zip(metrics, w7_8, gates)):
    if w7_val >= gate_val:
        ax.text(i, 1.08, '✓ PASS', ha='center', fontsize=10, 
                fontweight='bold', color='green')
    else:
        ax.text(i, 1.08, '✗ FAIL', ha='center', fontsize=10, 
                fontweight='bold', color='red')

plt.tight_layout()

# Save figure
output_path = 'reports/figures/srs_comparison_w5-6_vs_w7-8.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

# Also save as PDF for LaTeX
output_pdf = 'reports/figures/srs_comparison_w5-6_vs_w7-8.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✅ Saved: {output_pdf}")

# Display
plt.show()

print("\n📊 SRS Comparison Metrics:")
for metric, w5, w7, gate in zip(metrics, w5_6, w7_8, gates):
    delta = w7 - w5
    delta_pct = (delta / w5 * 100) if w5 > 0 else 0
    status = "✓ PASS" if w7 >= gate else "✗ FAIL"
    print(f"  {metric:5s}: {w5:.4f} → {w7:.4f} (Δ={delta:+.4f}, {delta_pct:+6.1f}%) [{status}]")
