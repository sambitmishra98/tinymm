## Progress on matrix multiplication on NVIDIA GPUs 

## Step 1: Basic C++ CUDA code

Location: `src/dense/cuda/minimal.cu`

## Step 1: Convert to row-major: C = A × B to Cᵀ = Aᵀ × Bᵀ

Location: `src/dense/cuda/transposed-minimal.cu`

## Step 2: Improve performance with k-split algorithms available on cuBLAS

