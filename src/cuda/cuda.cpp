// Example build command:
//   nvcc src/denseMM/cuda.cpp src/common/common.cpp -o executables/dense_cuda.exe -lcublas
//
// Usage:
//   ./executables/dense_cuda.exe <device> <mmtype> <matrix.mtx> <n> <niters>
// Example:
//   ./executables/dense_cuda.exe H100 dense operators/p3/hex/M0.mtx 1000 10

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "src/common/common.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]){ if (argc < 6) { cerr << "Usage: " << argv[0] << " <device> <mmtype> <matrix.mtx> <n> <niters>\n"; return 1; }

    // 1. Parse command-line
    string device = argv[1];     // e.g. "H100/A100/MAX1550/MI300x/MI200/MI100"
    string mmtype = argv[2];     // e.g. "dense/sparse/kernel"
    string mtxfile = argv[3];    // e.g. "operators/p{}/{}/M${}.mtx"
    size_t n = static_cast<size_t>(atoi(argv[4]));
    int niters = atoi(argv[5]);

    FileMetadata meta = parseFilename(mtxfile);
    meta.mmtype = mmtype;
    vector<double> A_data;
    size_t m, k;
    readMTXMatrix(mtxfile, A_data, m, k, meta);

    vector<double> B_data(k * n, 1.0); // B is k x n
    vector<double> C_data(m * n, 0.0); // C is m x n

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_data.size() * sizeof(double));
    cudaMalloc(&d_B, B_data.size() * sizeof(double));
    cudaMalloc(&d_C, C_data.size() * sizeof(double));

    cudaMemcpy(d_A, A_data.data(), A_data.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_data.data(), B_data.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_data.data(), C_data.size()*sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0, beta = 0.0;

    // Warm-up
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    cudaDeviceSynchronize();

    // 8. Timed loop
    auto start = high_resolution_clock::now();
    for(int i = 0; i < niters; i++){
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    }
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    double total = duration<double>(end - start).count();
    double avg = total / niters;

    double bw = processBandwidth(meta, m, k, n, avg);

    writeOutputCSV(device, meta, n, niters, total, bw);

    cublasDestroy(handle); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
