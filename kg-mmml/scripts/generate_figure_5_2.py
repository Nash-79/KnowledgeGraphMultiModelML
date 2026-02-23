"""Generate Figure 5.2: Latency Scaling Comparison"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

N = [3218, 10000, 32180, 100000]
annoy = [0.037, 0.045, 0.055, 0.068]
faiss = [0.042, 0.051, 0.063, 0.078]
filtered = [0.089, 0.280, 0.850, 2.650]
exact = [2.140, 6.650, 21.400, 66.500]

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(N, annoy, 'o-', color='#4472C4', linewidth=2, label='Annoy (20 trees)')
ax.loglog(N, faiss, 's-', color='#70AD47', linewidth=2, label='FAISS-HNSW (M=16)')
ax.loglog(N, filtered, '^--', color='#FFC000', linewidth=2, label='Filtered Cosine')
ax.loglog(N, exact, 'v:', color='#C00000', linewidth=2, label='Exact Cosine')
ax.axhline(y=150, color='#808080', linestyle='--', linewidth=1.5, label='SLO (150ms)')

ax.set_xlabel('Corpus Size (N)', fontsize=12)
ax.set_ylabel('p99 Latency (ms)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(2000, 150000)
ax.set_ylim(0.01, 200)

plt.tight_layout()
out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig_5_2_latency_scaling.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Figure 5.2 saved to: {out_path}")
