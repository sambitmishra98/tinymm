# tinymm-benchmarking

Benchmarking repository for double-precision GPU matrix multiplication (`C = A × B`) used in PyFR. Targets NVIDIA (cuBLAS), AMD (hipBLAS), and Intel (oneMKL via SYCL) GPUs. Operator matrices (`A`) vary in size (10s–100s), dense or sparse. Operand matrices (`B`) typically range from 10⁶–10⁷ rows.

## Structure
```
tinymm-benchmarking/
├── plots/                 # Visualization scripts/results
├── tools/                 # Data processing utilities
├── results/               # Benchmark outputs
│   ├── AMD/MI300x/bench_dense.csv
│   ├── Intel/MAX1550/
│   └── NVIDIA/
│       ├── A100/bench_dense.csv
│       └── H100/
├── samples/               # Input matrices by polynomial order (p${i})
│   └── pyfr/mats/p${i}/${etype}/${OPMAT}-${k}x${m}-{sp/de}.mtx
└── src/
    ├── dense/
    │   ├── common.{cpp,h}      # Common utilities (parsing, I/O)
    │   ├── cuda-effort.cu      # CUDA benchmarks
    │   └── rocm-minimal.hip    # HIPified CUDA
    ├── dense-unrolled/
    │   └── cuda-effort.cu      # Unrolled CUDA
    ├── gimmik/
    │   └── pyfr_gimmik_effort.py
    └── sparse/
        └── cuda-effort.cu
```

### Matrix Filename Convention
```
samples/pyfr/mats/p${i}/${etype}/${OPMAT}-${k}x${m}-{sp/de}.mtx
```

## Compilation & Execution

### NVIDIA GPUs (CUDA)
```bash
module load cuda/12.3.0
nvcc src/dense/cuda-effort.cu src/dense/common.cpp -o src/dense/cuda-effort.exe -lcublas

./src/dense/cuda-effort.exe samples/pyfr/mats/p0/hex/M0-6x1-sp.mtx 10000000 1000 NVIDIA A100
```

### AMD GPUs (HIP)
```bash
module load rocm/6.3.2 llvm # llvm for hipify-clang only
hipify-clang src/dense/cuda-effort.cu -o src/dense/rocm-effort.hip
hipcc src/dense/rocm-effort.hip src/dense/common.cpp -o src/dense/rocm-effort.exe -lrocblas

./src/dense/rocm-effort.exe samples/pyfr/mats/p0/hex/M0-6x1-sp.mtx 10000000 1000 AMD MI300x
```

### Intel GPUs (SYCL, oneMKL)
```bash
module load intel/oneapi/release/2024.1
icpx -fsycl -O3 -fsycl-device-code-split=per_source src/dense/mkl-effort.cpp src/dense/common.cpp -o src/dense/mkl-effort.exe -Wl,--start-group -lmkl_sycl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -liomp5 -lpthread -lm

./src/dense/mkl_effort.exe samples/pyfr/mats/p3/hex/M0-96x64-de.mtx 100 100 Intel MAX1550
```

## Automated Benchmarking

Example setup 
```sh
export TINYMM_VENDOR="NVIDIA"
export TINYMM_DEVICE="H100"
export TINYMM_BMATRIXLENGTH=1000000
export TINYMM_ITERATIONS=1000
export TINYMM_ETYPES="hex"  # pri, quad, tet, tri or all
export TINYMM_ORDERS="all"  # 0–8 or all

if [[ "$TINYMM_ETYPES" == "all" ]]; then TINYMM_ETYPES="hex pri quad tet tri"; fi
if [[ "$TINYMM_ORDERS" == "all" ]]; then TINYMM_ORDERS="0 1 2 3 4 5 6 7 8"; fi

# Fixing vendor selection and module loading
if   [[ "$TINYMM_VENDOR" == "NVIDIA" ]]; then export TINYMM_BE="cuda" ; module load cuda/12.3.0
elif [[ "$TINYMM_VENDOR" == "AMD"    ]]; then export TINYMM_BE="rocm" ; module load rocm/6.3.2 
elif [[ "$TINYMM_VENDOR" == "Intel"  ]]; then export TINYMM_BE="mkl"  ; module load intel/oneapi/release/2024.1
fi

# Ensure results directory exists
mkdir -p "results/$TINYMM_VENDOR/$TINYMM_DEVICE"

```

Benchmark dense MM kernels:

```sh
for ORDER in $TINYMM_ORDERS; do
  for ETYPE in $TINYMM_ETYPES; do
    for MATRIX in $(ls samples/pyfr/mats/p${ORDER}/${ETYPE}/); do
      if [[ $MATRIX != *-de.mtx ]]; then continue; fi
      echo "$TINYMM_VENDOR $TINYMM_DEVICE Order:$ORDER Type:$ETYPE Matrix:$MATRIX"
      ./src/dense/${TINYMM_BE}-minimal.exe samples/pyfr/mats/p${ORDER}/${ETYPE}/$MATRIX $TINYMM_BMATRIXLENGTH $TINYMM_ITERATIONS $TINYMM_VENDOR $TINYMM_DEVICE
      sleep 0.1
    done
  done
done
```

Benchmark sparse MM kernels:

```sh
for ORDER in $TINYMM_ORDERS; do
  for ETYPE in $TINYMM_ETYPES; do
    for MATRIX in $(ls samples/pyfr/mats/p${ORDER}/${ETYPE}/); do
      if [[ $MATRIX != *-sp.mtx ]]; then continue; fi
      echo "$TINYMM_VENDOR $TINYMM_DEVICE Order:$ORDER Type:$ETYPE Matrix:$MATRIX"
      ./src/sparse/${TINYMM_BE}-effort.exe samples/pyfr/mats/p${ORDER}/${ETYPE}/$MATRIX $TINYMM_BMATRIXLENGTH $TINYMM_ITERATIONS $TINYMM_VENDOR $TINYMM_DEVICE
      sleep 0.1
    done
  done
done
```