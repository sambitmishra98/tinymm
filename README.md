# tinymm-benchmarking

This repository benchmarks the performance of double-precision matrix multiplication (`C = A × B`) operations used in PyFR, focusing on GPU platforms (NVIDIA, AMD, Intel). Operator matrix (`A`) sizes range from tens to hundreds, typically dense or sparse, whereas the operand matrix (`B`) has dimensions on the order of 10⁶–10⁷ rows.

## Directory Structure

```
tinymm-benchmarking/
├── plots/                 # Visualization scripts and benchmark plots
├── tools/                 # Utility scripts for data conversion/preprocessing
├── results/               # CSV benchmark outputs per hardware platform
│   ├── AMD/MI300x/bench_dense.csv
│   ├── Intel/MAX-1550/
│   └── NVIDIA/
│       ├── A100/bench_dense.csv
│       └── H100/
├── samples/               # Input matrices organized by polynomial order (p${i})
│   └── pyfr/mats/p${i}/${etype}/${OPMAT}-${k}x${m}-{sp/de}.mtx
└── src/
    ├── dense/             # Dense matrix multiplication benchmarks
    │   ├── common.cpp     # Common utility functions for parsing and I/O
    │   ├── common.h
    │   ├── cuda-effort.cu # CUDA implementation
    │   └── rocm-minimal.hip # HIP implementation converted from CUDA
    ├── dense-unrolled/
    │   └── cuda-effort.cu # Unrolled dense CUDA benchmarks
    ├── gimmik/
    │   └── pyfr_gimmik_effort.py # GiMMiK benchmarking script
    └── sparse/
        └── cuda-effort.cu # Sparse CUDA benchmarks
```

### Input Matrix Naming Convention

Operator matrices should adhere to:
```
samples/pyfr/mats/p${i}/${etype}/${OPMAT}-${k}x${m}-{sp/de}.mtx
```
- `${etype}`: Element type (e.g., hex, pri, quad, tet, tri)
- `{sp/de}`: Indicates sparse (`sp`) or dense (`de`) format

## Running Benchmarks

### NVIDIA GPUs (cuBLAS)

```bash
module load cuda/12.3.0
nvcc src/dense/cuda-effort.cu src/dense/common.cpp -o src/dense/cuda-effort.exe -lcublas

# Execute benchmark
./src/dense/cuda-effort.exe samples/pyfr/mats/p0/hex/M0-6x1-sp.mtx 10000000 1000
```

### AMD GPUs (hipBLAS)

```bash
module load rocm/6.3.2 llvm
hipify-clang src/dense/cuda-effort.cu -o src/dense/rocm-effort.hip
hipcc src/dense/rocm-effort.hip src/dense/common.cpp -o src/dense/rocm-effort.exe -lrocblas

# Execute benchmark
./src/dense/rocm-effort.exe samples/pyfr/mats/p0/hex/M0-6x1-sp.mtx 10000000 1000
```

## Automated Benchmark Runs

Example automation loop:
```bash
for SIZE in 100 1000 10000 100000 1000000 10000000; do
    ./src/dense/cuda-effort.exe samples/pyfr/mats/p0/hex/M0-6x1-sp.mtx $SIZE 1000
    # For AMD, replace executable with rocm-effort.exe
    sleep 1
    nvidia-smi # or rocm-smi
    sleep 1
done
```

Benchmark results are appended automatically to CSV files under `results/`.

