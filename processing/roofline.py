import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

VENDOR="Intel"
DEVICE="MAX1550"
PRECISION="double"

# --- Set Seaborn theme ---
sns.set_theme(style="whitegrid")

# --- Define file paths ---
csv_file = f"results/{VENDOR}/{DEVICE}/bench_de.csv"
output_dir = "plots/"
output_file = os.path.join(output_dir, f"roofline-{VENDOR}-{DEVICE}-{PRECISION}.png")

# --- Read CSV data ---
df = pd.read_csv(csv_file)

# Extract only the required columns: order, BW, AI.
orders = df["order"].astype(int).values
Cm_measured = df["Cm"].values # Convert BW to GFLOPS (if originally in FLOPs)
AI_measured = df["AI"].values

# --- Theoretical Roofline Parameters ---
theo_bw = 1700 * 1e9   # Theoretical memory bandwidth in GB/s (interpreted as GFLOPS per unit AI)
theo_peak = 51.2 * 1e12   # Peak performance in GFLOPS (52.43 TFLOPS)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot scatter points, grouping by order
for order in np.unique(orders):
    mask = orders == order
    ax.scatter(AI_measured[mask], Cm_measured[mask],
               s=80, color='tab:blue',
                 alpha=0.1 * (order+2),
               label=f'p{order} HEX', edgecolors='k', zorder=3)

# Create a smooth range for AI (x-axis) for plotting roofline curves.
AI_range = np.logspace(np.log10(min(AI_measured)*0.8), np.log10(max(AI_measured)*2), 200)

# Memory-bound roofline: performance = AI * theo_bw.
mem_bound = AI_range * theo_bw

# Compute-bound roofline is horizontal at theo_peak.
compute_bound = np.full_like(AI_range, theo_peak)

# Plot a line segment for the roofline curve, instead of a long line
ax.plot(AI_range,     mem_bound, linestyle='--', color='k', linewidth=2, label=f'{VENDOR} {DEVICE} {PRECISION}: {theo_bw   /1e9:.2f} GB/s')
ax.plot(AI_range, compute_bound, linestyle='--', color='r', linewidth=2, label=f'{VENDOR} {DEVICE} {PRECISION}: {theo_peak/1e12:.2f} TFLOPS')

# Set log scale for both axes (common for roofline plots).
ax.set_xscale('log')
ax.set_yscale('log')

# Set axis labels and title.
ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
ax.set_ylabel('Performance (FLOPS)', fontsize=12)
ax.set_title('Roofline Model for Intel MAX 1550 GPU, double precision', fontsize=14)

# Add legend and grid for clarity.
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()

# Save the plot to the designated output directory.
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
