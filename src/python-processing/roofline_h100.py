import pandas as pd

df = pd.read_csv("/home/sambit.mishra/scratch/03_KERNELPERFORMANCE/tinymm-benchmarking/results/benchmarks.csv")

# Suppose you have a function or formula that returns the FLOP count
def estimate_flops(row):
    # row["AMatSize"] holds (m*k) for dense
    # row["AMatnnz"]  holds nnz for sparse
    # row["n"]        is how many times we repeated the operation
    mmtype = row["mmtype"]
    
    if mmtype == "dense":
        # 2 flops per entry (multiply + add) * AMatSize * n
        return 2.0 * row["AMatSize"] * row["n"]
    else:
        # 2 flops per non-zero * AMatnnz * n
        return 2.0 * row["AMatnnz"] * row["n"]

# FLOPS is 


df["FLOPs"] = df.apply(estimate_flops, axis=1)  # or read from a column if you have it

# Bytes transferred (this is the measured traffic)
df["Bytes"] = df["BW"] * df["wtime"]

# Arithmetic Intensity
df["AI"] = df["FLOPs"] / df["Bytes"]

# Performance (FLOP/s)
df["Perf"] = df["FLOPs"] / df["wtime"]

# Now you can plot (AI, Perf) with a log-log scale, for example with matplotlib:
import matplotlib.pyplot as plt

plt.xscale("log")
plt.yscale("log")
plt.scatter(df["AI"], df["Perf"])

# Memory roof line:
peak_bw = 2.0e12  # e.g. 2 TB/s for H100
ai_vals = [1e-1, 1e2]  # spans some range of AI
roof_mem = [ai * peak_bw for ai in ai_vals]
plt.plot(ai_vals, roof_mem, label="Memory roof")

# Compute roof line:
peak_flops = 60e12  # e.g. 60 TF/s
plt.hlines(peak_flops, xmin=1e-2, xmax=1e6, label="Compute roof")

plt.xlabel("Arithmetic Intensity [FLOPs/Byte]")
plt.ylabel("Performance [FLOPs/s]")
plt.legend()
plt.show()

# Save the plot 
plt.savefig("results/roofline_h100.png")