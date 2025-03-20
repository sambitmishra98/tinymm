// Location: "src/cuda/sparse.cpp"
// nvcc -O3 
//      -Wno-deprecated-declarations src/cuda/sparse.cpp src/base.cpp src/cuda/base.cpp 
//      -x cu 
//      -o src/cuda/sparse.exe -lcusparse -lcublas
// ./src/cuda/sparse.exe h100 sparse operators/p3/hex/M0.mtx 1000000 5

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include "src/base.h"       // parseCmdLineArgs, readMTXdense, parseFilename, etc.
#include "src/cuda/base.h"  // cudaAllocDouble, cudaCopyToDevice, cudaCopyToHost, etc.

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[])
{
    // 1. Parse command-line (similar to dense)
    CmdLineArgs args = parseCmdLineArgs(argc, argv);

    // 2. Parse metadata from path
    FileMetadata meta = parseFilename(args.matrixPath);
    meta.mmtype = args.mmtype; // e.g. "sparse"
    meta.backend = "direct";

    // 3. Read matrix A in dense form first (so we can do a reference check)
    vector<double> A_data;
    size_t m, k;
    readMTXdense(args.matrixPath, A_data, m, k, meta); // dense, col-major
    int nnz = meta.nnz;  // number of nonzeros from the .mtx

    // ---- Convert dense => CSR ----
    //    We'll do a 2-pass approach to fill rowPtr, colInd, vals
    vector<int> hRowPtr(m+1, 0);
    vector<int> hColInd(nnz);
    vector<double> hVals(nnz);

    // pass 1: count per row
    for (size_t col = 0; col < k; col++) {
        for (size_t row = 0; row < m; row++) {
            double val = A_data[col*m + row];
            if (val != 0.0) {
                hRowPtr[row+1]++;
            }
        }
    }
    for (size_t r = 0; r < m; r++) {
        hRowPtr[r+1] += hRowPtr[r];
    }

    // pass 2: fill colInd, vals
    vector<int> rowStart = hRowPtr;
    for (size_t col = 0; col < k; col++) {
        for (size_t row = 0; row < m; row++) {
            double val = A_data[col*m + row];
            if (val != 0.0) {
                int dest = rowStart[row]++;
                hColInd[dest] = (int)col;
                hVals[dest]   = val;
            }
        }
    }

    // 4. Prepare B, C on host (like in dense.cpp)
    //    B is k x n (all ones), C is m x n (zeros)
    vector<double> B_host(k * args.n, 1.0);
    vector<double> C_host(m * args.n, 0.0);

    // 5. Allocate device memory for the CSR, B, C
    int *dRowPtr = nullptr, *dColInd = nullptr;
    double *dVals = nullptr, *dB = nullptr, *dC = nullptr;

    cudaMalloc(&dRowPtr, (m+1)*sizeof(int));
    cudaMalloc(&dColInd, nnz*sizeof(int));
    cudaMalloc(&dVals,   nnz*sizeof(double));

    cudaAllocDouble(&dB, B_host.size()); // actually (k*n)
    cudaAllocDouble(&dC, C_host.size()); // (m*n)

    // 6. Copy host->device
    cudaMemcpy(dRowPtr,  hRowPtr.data(),  (m+1)*sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(dColInd,  hColInd.data(),  nnz*sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(dVals,    hVals.data(),    nnz*sizeof(double),   cudaMemcpyHostToDevice);

    cudaCopyToDevice(dB, B_host, B_host.size());
    cudaCopyToDevice(dC, C_host, C_host.size());

    // 7. Create cuSPARSE handle
    cusparseHandle_t spHandle;
    cusparseCreate(&spHandle);

    // 8. Create the SpMat and DnMat descriptors
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA,
        (int)m, (int)k, nnz,
        dRowPtr, dColInd, dVals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnMatDescr_t matB, matC;
    // B is k x n, col-major
    cusparseCreateDnMat(&matB, (int)k, (int)args.n, (int)k,
                        dB, CUDA_R_64F, CUSPARSE_ORDER_COL);
    // C is m x n, col-major
    cusparseCreateDnMat(&matC, (int)m, (int)args.n, (int)m,
                        dC, CUDA_R_64F, CUSPARSE_ORDER_COL);

    double alpha = 1.0, beta = 0.0;

    // 9. Setup SpMM buffer
    cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    cusparseSpMM_bufferSize(spHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_64F, alg, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // -------- Warm-up: spMM => dC --------
    cusparseSpMM(spHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_64F, alg, dBuffer);
    cudaDeviceSynchronize();

    // -------- Correctness check vs. dense reference --------
    // We'll reuse the dense array A_data on the device. We need a separate device pointer dA_dense:
    double *dA_dense = nullptr;
    cudaAllocDouble(&dA_dense, A_data.size());
    cudaCopyToDevice(dA_dense, A_data, A_data.size());

    // We'll also allocate a reference pointer dR. We'll do cublasDgemm => dR, then compare to dC.
    double *dR = nullptr;
    cudaAllocDouble(&dR, C_host.size());  // m*n
    cudaMemset(dR, 0, C_host.size()*sizeof(double));

    // Create a cuBLAS handle for reference multiply
    cublasHandle_t blasHandle;
    cublasCreate(&blasHandle);

    // dR = A * B  (dense) => reference
    // A is (m x k), B is (k x n), col-major
    // => cublasDgemm uses leading dims: A => m, B => k, C => m
    cublasDgemm(blasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                (int)m, (int)args.n, (int)k,
                &alpha,
                dA_dense, (int)m,
                dB,       (int)k,
                &beta,
                dR,       (int)m);
    cudaDeviceSynchronize();

    // Now copy dC and dR back to host to compare
    vector<double> sparseC(m*args.n), refC(m*args.n);
    cudaCopyToHost(sparseC, dC, sparseC.size());
    cudaCopyToHost(refC,    dR, refC.size());

    bool match = true;
    for (size_t i = 0; i < sparseC.size(); i++) { 
        double diff = fabs(sparseC[i] - refC[i]);
        if (diff > 1e-6) {
            match = false;
            break;
        }
    }
    if (!match) { cerr << "MISMATCH !!! cuSPARSE result ≠ cuBLAS reference.\n"; return 1; }

    // ------ Timed loop for SpMM ------
    auto start = high_resolution_clock::now();
    for(int i = 0; i < args.niters; i++){
        cusparseSpMM(spHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_64F, alg, dBuffer);
    }
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    double avg = duration<double>(end - start).count() / args.niters;
    writeOutputCSV(args.device, meta, args.n, avg, efficiency(meta, m, k, args.n, avg));

    // ------ Cleanup ------
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(spHandle);

    cublasDestroy(blasHandle);

    cudaFree(dRowPtr);
    cudaFree(dColInd);
    cudaFree(dVals);
    cudaFree(dB);
    cudaFree(dC);

    cudaFreeDouble(dA_dense);
    cudaFreeDouble(dR);

    return 0;
}
