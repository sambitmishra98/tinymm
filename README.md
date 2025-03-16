# Tiny Matrix Multiplication

This repository solely focusses on matrix multiplications of a particular form (`C = A × B`) where operator matrices `A` have 10s to 100s of rows/columns, and operand matrices `B` range from 10⁶–10⁷ rows. 
Performance of these matrices are benchmarked with vendor-provided libraries on GPUs. 
The structure of these matrices allows various optimisation techniques to be applied, such as unrolling the operator matrices and cache-blocking etc. 
Our goal is to strive for maximum possible performance of these specific matrix operations on the GPUs.

### Matrix Filename Convention
```
samples/pyfr/mats/p${i}/${etype}/${OPMAT}.mtx
```

## Compilation & Execution

### NVIDIA GPUs (CUDA)
```bash
module load cuda/12.3.0
nvcc src/dense/cuda/minimal.cu src/common.cpp -o src/dense/cuda/minimal.exe -lcublas

./src/dense/cuda/minimal.exe samples/pyfr/mats/p0/hex/M0-6x1-sp.mtx 10000000 1000 NVIDIA A100
```

### AMD GPUs (HIP)
```bash
module load rocm/6.3.2 llvm # llvm for hipify-clang only
hipify-clang src/dense/cuda/minimal.cu -o src/dense/rocm/minimal.hip
hipcc src/dense/rocm/minimal.hip src/common.cpp -o src/dense/rocm/minimal.exe -lrocblas

./src/dense/rocm/minimal.exe samples/pyfr/mats/p0/hex/M0-6x1-sp.mtx 10000000 1000 AMD MI300x
```

### Intel GPUs (SYCL, oneMKL)
```bash
module load intel/oneapi/release/2024.1
icpx -fsycl -O3 -fsycl-device-code-split=per_source src/dense/mkl/minimal.cpp src/common.cpp -o src/dense/mkl/minimal.exe -Wl,--start-group -lmkl_sycl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -liomp5 -lpthread -lm

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
      ./src/sparse/${TINYMM_BE}-minimal.exe samples/pyfr/mats/p${ORDER}/${ETYPE}/$MATRIX $TINYMM_BMATRIXLENGTH $TINYMM_ITERATIONS $TINYMM_VENDOR $TINYMM_DEVICE
      sleep 0.1
    done
  done
done
```