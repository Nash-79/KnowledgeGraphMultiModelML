"""Generate Figure 5.1: SRS Component Comparison"""
import matplotlib.pyplot as plt
from pathlib import Path

components = ['HP', 'AtP', 'AP', 'RTF']
scores = [0.2726, 0.9987, 1.0000, 1.0000]
colours = ['#4472C4', '#70AD47', '#FFC000', '#7030A0']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(components, scores, color=colours, alpha=0.8)
ax.axvline(x=0.8179, color='#C00000', linestyle='--', linewidth=2, label='SRS=0.8179')
ax.set_xlabel('Score', fontsize=12)
ax.set_ylabel('Component', fontsize=12)
ax.set_xlim(0, 1.05)
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(score + 0.02, i, f'{score:.4f}', va='center', fontsize=10)

plt.tight_layout()
out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig_5_1_srs_components.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Figure 5.1 saved to: {out_path}")
