"""Generate Figure 5.5: Error Distribution by Category"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

categories = ['Other', 'Revenue', 'Liabilities', 'Expenses', 'Assets', 'Equity', 'Income']
baseline_errors = [3.05, 1.37, 0.86, 0.52, 0.00, 0.00, 0.00]
text_concept_errors = [0.68, 0.19, 0.43, 0.26, 0.00, 0.00, 0.00]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, baseline_errors, width, label='Baseline (text-only)',
               color='#FFC000', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, text_concept_errors, width, label='Text+Concept',
               color='#4472C4', alpha=0.8, edgecolor='black')

ax.set_ylabel('Error Rate (%)', fontsize=12)
ax.set_xlabel('Financial Statement Category', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 3.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig_5_5_error_distribution.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Figure 5.5 saved to: {out_path}")
