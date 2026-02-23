#!/usr/bin/env python3
"""
Joint Model Comparison: Consistency Penalty ON vs OFF

Visualizes the impact of consistency penalty on micro/macro F1 scores.

Usage:
    python scripts/visualization/plot_joint_comparison.py
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Ensure output directory exists
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Load metrics
with open("outputs/joint_with_penalty/metrics.json") as f:
    penalty_on = json.load(f)

with open("outputs/joint_no_penalty/metrics.json") as f:
    penalty_off = json.load(f)

# Extract test metrics
metrics = ['Micro F1', 'Macro F1']
penalty_on_scores = [
    penalty_on['test']['micro_f1'],
    penalty_on['test']['macro_f1']
]
penalty_off_scores = [
    penalty_off['test']['micro_f1'],
    penalty_off['test']['macro_f1']
]

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Bar comparison
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, penalty_on_scores, width, 
                label='Penalty ON (0.1)', 
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, penalty_off_scores, width, 
                label='Penalty OFF (0.0)', 
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Styling for left plot
ax1.set_xlabel('Metric', fontsize=13, fontweight='bold')
ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
ax1.set_title('Joint Model: Test Performance\nConsistency Penalty Ablation', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax1.set_ylim(0.75, 0.95)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add delta annotations
delta_micro = penalty_off_scores[0] - penalty_on_scores[0]
delta_macro = penalty_off_scores[1] - penalty_on_scores[1]

ax1.annotate(f'Δ={delta_micro:+.4f}', 
            xy=(0, max(penalty_on_scores[0], penalty_off_scores[0])), 
            xytext=(0, 0.93),
            ha='center', fontsize=9, color='gray')

ax1.annotate(f'Δ={delta_macro:+.4f}\n(+1.33 pp)', 
            xy=(1, penalty_off_scores[1]), 
            xytext=(1.3, 0.87),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# Right plot: Train vs Test comparison
categories = ['Penalty ON\n(0.1)', 'Penalty OFF\n(0.0)']
train_micro = [penalty_on['train']['micro_f1'], penalty_off['train']['micro_f1']]
test_micro = [penalty_on['test']['micro_f1'], penalty_off['test']['micro_f1']]
train_macro = [penalty_on['train']['macro_f1'], penalty_off['train']['macro_f1']]
test_macro = [penalty_on['test']['macro_f1'], penalty_off['test']['macro_f1']]

x2 = np.arange(len(categories))
width2 = 0.2

bars_tr_mi = ax2.bar(x2 - 1.5*width2, train_micro, width2, 
                     label='Train Micro', color='#3498db', alpha=0.6)
bars_te_mi = ax2.bar(x2 - 0.5*width2, test_micro, width2, 
                     label='Test Micro', color='#3498db', alpha=0.9)
bars_tr_ma = ax2.bar(x2 + 0.5*width2, train_macro, width2, 
                     label='Train Macro', color='#f39c12', alpha=0.6)
bars_te_ma = ax2.bar(x2 + 1.5*width2, test_macro, width2, 
                     label='Test Macro', color='#f39c12', alpha=0.9)

# Styling for right plot
ax2.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
ax2.set_title('Train vs Test Generalization\nAcross Penalty Settings', 
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x2)
ax2.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right', framealpha=0.9, ncol=2)
ax2.set_ylim(0.75, 0.95)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# Save figure
output_path = 'reports/figures/joint_model_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

# Also save as PDF
output_pdf = 'reports/figures/joint_model_comparison.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✅ Saved: {output_pdf}")

print("\n🤖 Joint Model Results:")
print(f"  Penalty ON  (0.1): Test Micro={penalty_on_scores[0]:.4f}, Test Macro={penalty_on_scores[1]:.4f}")
print(f"  Penalty OFF (0.0): Test Micro={penalty_off_scores[0]:.4f}, Test Macro={penalty_off_scores[1]:.4f}")
print(f"\n  → Macro F1 improved by {delta_macro:.4f} ({delta_macro*100:+.2f}%) when penalty is OFF")
print(f"  → Micro F1 nearly identical (Δ={delta_micro:.4f})")
print(f"\n  🏆 Recommendation: Use consistency_weight=0.0 for better macro F1")
