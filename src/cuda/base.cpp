// Location: src/cuda/base.cpp

#include "src/cuda/base.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>    // for std::abs


void cudaAllocDouble(double** dPtr, size_t n)
{
    cudaMalloc(dPtr, n * sizeof(double));
}

void cudaCopyToDevice(double* dPtr,
                      const std::vector<double>& hData,
                      size_t n)
{
    cudaMemcpy(dPtr, hData.data(),
               n * sizeof(double),
               cudaMemcpyHostToDevice);
}

void cudaCopyToHost(std::vector<double>& hData,
                    const double* dPtr,
                    size_t n)
{
    cudaMemcpy(hData.data(),
               dPtr,
               n * sizeof(double),
               cudaMemcpyDeviceToHost);
}

void cudaFreeDouble(double* dPtr)
{
    if (dPtr) {
        cudaFree(dPtr);
    }
}


bool checkCublasGemmCorrectness(cublasHandle_t handle,
    size_t m, size_t n, size_t k,
    const double* dA,
    const double* dB,
    const double* dC,
    double alpha,
    double beta,
    double tol)
{
// 1) Allocate dR
size_t lenC = m * n;
double* dR = nullptr;
cudaAllocDouble(&dR, lenC);

// 2) Zero-initialize dR (or the existing contents) before calling gemm
cudaMemset(dR, 0, lenC*sizeof(double));

// 3) Do cublas gemm: R = alpha*A*B + beta*R
cublasStatus_t stat = cublasDgemm(handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          (int)m, (int)n, (int)k,
          &alpha,
          dA, (int)m,
          dB, (int)k,
          &beta,
          dR, (int)m);
if (stat != CUBLAS_STATUS_SUCCESS) {
std::cerr << "checkCublasGemmCorrectness: cublasDgemm failed with code "
<< stat << std::endl;
cudaFreeDouble(dR);
return false;
}
cudaDeviceSynchronize();

// 4) Copy both dC and dR back to host
std::vector<double> C_host(lenC), R_host(lenC);
cudaCopyToHost(C_host, dC, lenC);
cudaCopyToHost(R_host, dR, lenC);

// 5) Compare
bool match = true;
for (size_t i = 0; i < lenC; i++) {
double diff = std::abs(R_host[i] - C_host[i]);
if (diff > tol) {
match = false;
break;
}
}

// 6) If mismatch, print the first 10 elements
if (!match) {
std::cerr << "[checkCublasGemmCorrectness] Mismatch found!\n";
std::cerr << "First 10 elements from test (C) vs. reference (R):\n";
for (size_t i = 0; i < 10 && i < lenC; i++) {
double diff = std::abs(R_host[i] - C_host[i]);
std::cerr << "  idx " << i
<< ": C=" << C_host[i]
<< ", R=" << R_host[i]
<< ", diff=" << diff << "\n";
}
}

// 7) Free dR
cudaFreeDouble(dR);

return match;
}