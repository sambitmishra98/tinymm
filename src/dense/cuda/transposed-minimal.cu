#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "src/common.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace chrono;

// Helper function to transpose matrix
void transposeMatrix(const vector<double> &input, vector<double> &output, size_t rows, size_t cols) {
    for(size_t i = 0; i < rows; ++i)
        for(size_t j = 0; j < cols; ++j)
            output[j * rows + i] = input[i * cols + j];
}

int main(int argc, char* argv[]){
    FileMetadata meta = parseFilename(argv[1]);
    size_t n = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    string vendor = argv[4], device = argv[5];

    vector<double> A_data;
    size_t m, k;

    readMTXMatrix(argv[1], A_data, m, k, meta);

    vector<double> A_transposed(m * k);
    transposeMatrix(A_data, A_transposed, m, k);

    vector<double> B_data(k * n, 1.0); // B initialized with ones
    vector<double> B_transposed(n * k);
    transposeMatrix(B_data, B_transposed, k, n);

    vector<double> C_data(m * n, 0.0); // C initialized with zeros

    double *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, A_transposed.size() * sizeof(double));
    cudaMalloc(&d_B, B_transposed.size() * sizeof(double));
    cudaMalloc(&d_C, C_data.size() * sizeof(double));

    cudaMemcpy(d_A, A_transposed.data(), A_transposed.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_transposed.data(), B_transposed.size() * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle; cublasCreate(&handle);
    double alpha = 1.0, beta = 0.0;

    cudaDeviceSynchronize();

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, 
                                                                   d_B, k, 
                                                           &beta , d_C, m);
    cudaDeviceSynchronize();
    auto start = high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, 
                                                                       d_B, k, 
                                                                &beta, d_C, m);
    }

    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();

    double total = duration<double>(end - start).count();
    double avg = total / iterations;

    // Print time taken 
    cout << endl;
    cout << "Time taken: " << total << "s" << endl;

    writeOutputCSV(meta, n, m, k, iterations, avg, vendor, device);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}