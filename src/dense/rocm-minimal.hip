#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "common.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]){
    FileMetadata meta = parseFilename(argv[1]);
    size_t n = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    string vendor = argv[4], device = argv[5];
    vector<double> A_data;
    size_t m, k;

    readMTXMatrix(argv[1], A_data, m, k);

    vector<double> B_data(k * n, 1.0); // B is a matrix of all ones
    vector<double> C_data(m * n, 0.0); // C is a matrix of all zeros
    double *d_A, *d_B, *d_C;

    // Allocate device memory
    hipMalloc(&d_A, A_data.size() * sizeof(double));
    hipMalloc(&d_B, B_data.size() * sizeof(double));
    hipMalloc(&d_C, C_data.size() * sizeof(double));

    // Copy host data to device
    hipMemcpy(d_A, A_data.data(), A_data.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B_data.data(), B_data.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_C, C_data.data(), C_data.size() * sizeof(double), hipMemcpyHostToDevice);

    hipblasHandle_t handle; hipblasCreate(&handle);
    double alpha = 1.0, beta = 0.0;

    //   d_C = alpha *   d_A  *  d_B  + beta *  d_C
    // m x n =          m x k * k x n +        m x n
    hipblasDgemm(handle, 
        HIPBLAS_OP_N, HIPBLAS_OP_N, 
        m, n, k, 
        &alpha, d_A, m, 
                d_B, k, 
        &beta , d_C, m);
    hipDeviceSynchronize();
    auto start = high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        //   d_C = alpha *   d_A  *  d_B  + beta *  d_C
        // m x n =          m x k * k x n +        m x n
        hipblasDgemm(handle, 
            HIPBLAS_OP_N, HIPBLAS_OP_N, 
            m, n, k, 
            &alpha, d_A, m, 
            d_B, k, 
            &beta, d_C, m);
    }

    hipDeviceSynchronize();
    auto end = high_resolution_clock::now();
    double total = duration<double>(end - start).count();
    double avg = total / iterations;
    writeOutputCSV(meta, n, m, k, iterations, avg, vendor, device);
    hipblasDestroy(handle);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}
