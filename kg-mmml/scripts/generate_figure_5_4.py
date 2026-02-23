"""Generate Figure 5.4: Robustness Under Perturbation"""
import matplotlib.pyplot as plt
from pathlib import Path

scenarios = ['Baseline', 'Taxonomy\nRemoval', '5% Unit\nNoise', '10% Unit\nNoise']
srs_scores = [0.8179, 0.6642, 0.7607, 0.7058]
deltas = [0, -0.1537, -0.0572, -0.1121]
colours = ['#70AD47', '#C00000', '#FFC000', '#ED7D31']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(scenarios, srs_scores, color=colours, alpha=0.8, edgecolor='black')
ax.axhline(y=0.75, color='#808080', linestyle='--', linewidth=1.5, label='SRS Threshold (0.75)')

ax.set_ylabel('Semantic Retention Score', fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels and delta annotations
for i, (bar, score, delta) in enumerate(zip(bars, srs_scores, deltas)):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.02,
            f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if delta < 0:
        ax.text(bar.get_x() + bar.get_width()/2, score - 0.05,
                f'({delta:.1%})', ha='center', va='top', fontsize=9, color='darkred')

plt.tight_layout()
out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig_5_4_robustness_perturbation.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Figure 5.4 saved to: {out_path}")
