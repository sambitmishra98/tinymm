#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "src/base.h"         // parseCmdLineArgs, readMTXdense, etc.
#include "src/cuda/base.h"    // cudaAllocDouble, cudaCopyToHost, etc.

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[])
{
    CmdLineArgs args = parseCmdLineArgs(argc, argv);
    FileMetadata meta = parseFilename(args.matrixPath);
    meta.mmtype = args.mmtype;

    // 3. Read A
    vector<double> A_data; size_t m, k; readMTXdense(args.matrixPath, A_data, m, k, meta);

    // 4. Prepare B, C on host
    //    B is k x n, all ones
    //    C is m x n, all zeros
    vector<double> B_data(k * args.n, 1.0);
    vector<double> C_data(m * args.n, 0.0);
    
    // 5. Allocate device memory
    double *dA=nullptr, *dB=nullptr, *dC=nullptr;
    
    cudaAllocDouble(&dA, A_data.size());
    cudaAllocDouble(&dB, B_data.size());
    cudaAllocDouble(&dC, C_data.size());
    
    // 6. Copy host -> device
    cudaCopyToDevice(dA, A_data, A_data.size());
    cudaCopyToDevice(dB, B_data, B_data.size());
    cudaCopyToDevice(dC, C_data, C_data.size());

    // 7. Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    double alpha = 1.0, beta = 0.0;
    
    // 8. Warm-up
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)args.n, (int)k,
                &alpha, dA, (int)m, dB, (int)k, &beta, dC, (int)m);
        cudaDeviceSynchronize();
        
    // 8. Check correctness: compare dC vs. reference (computed internally)
    bool ok = checkCublasGemmCorrectness(handle, m, args.n, k, dA, dB, dC, alpha, beta, 1e-6);
    if (!ok) { cerr << "[Warm-Up] Results do not match reference.\n"; return 1; }

    // 9. Timed loop
    auto start = high_resolution_clock::now();
    for(int i = 0; i < args.niters; i++)
    {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)args.n, (int)k, 
                    &alpha, dA, (int)m, dB, (int)k, &beta, dC, (int)m);
    }                
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();

    double avg = duration<double>(end - start).count() / args.niters;
    double bw  = processBandwidth(meta, m, k, args.n, avg);
    writeOutputCSV(args.device, meta, args.n, avg, bw);

    // Cleanup
    cublasDestroy(handle);
    cudaFreeDouble(dA);
    cudaFreeDouble(dB);
    cudaFreeDouble(dC);

    return 0;
}
