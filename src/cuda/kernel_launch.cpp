// Location: src/cuda/kernel_launch.cpp

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include "src/base.h"         // parseCmdLineArgs, readMTXdense, etc.
#include "src/cuda/base.h"    // cudaAllocDouble, cudaCopyToDevice, cudaCopyToHost, etc.

using namespace std;
using namespace std::chrono;

// The GiMMiK kernel is declared as an extern symbol so the linker knows about it.
// The actual definition is in, e.g., "kernels/gimmik/cuda/p3/hex/M0.cpp"
extern __global__ void gimmik_mm(int n,
                                 const double* __restrict__ b, int ldb,
                                 double* __restrict__ c, int ldc);

/**
 * \brief Parse the kernel path, e.g. "kernels/gimmik/cuda/p3/hex/M0.cpp"
 *        to build a matching .mtx path "operators/p3/hex/M0.mtx".
 *        Similar to parseFilename, but we parse from kernel path.
 *
 * \param kernelPath (in) path to the GiMMiK kernel .cpp file
 * \param meta       (out) partially filled FileMetadata (order, etype, AMatName, mmtype)
 * \return           the derived .mtx path, e.g. "operators/p3/hex/M0.mtx"
 */
std::string kernelToMtxPath(const std::string &kernelPath, FileMetadata &meta)
{
    // Example: kernelPath = "kernels/gimmik/cuda/p3/hex/M0_bstream.cpp"
    // 1) Find "kernels/gimmik/p"
    size_t pos = kernelPath.find("kernels/gimmik/p");
    if (pos == std::string::npos) {
        // Fallback if not found
        return "";
    }
    // Advance past "kernels/gimmik/"
    pos += std::string("kernels/gimmik/").size();

    // Next char => polynomial order, e.g. "3"
    // So "p3" => we skip 'p'
    pos++; // skip 'p'
    meta.order = kernelPath.substr(pos, 1);

    // Next slash => e.g. "...p3/hex..."
    size_t slashPos = kernelPath.find('/', pos);
    // Extract up to 3 letters for etype, e.g. "hex"
    meta.etype = kernelPath.substr(slashPos + 1, 3);

    // Final filename => e.g. "M0_bstream.cpp"
    std::string fname = kernelPath.substr(kernelPath.find_last_of('/') + 1);
    // Remove ".cpp"
    size_t dotPos = fname.rfind(".cpp");
    std::string baseName = (dotPos != std::string::npos)
                           ? fname.substr(0, dotPos)
                           : fname;

    // There will always be three words separated by underscore
    // e.g. "M0_cuda_bstream-ksplit" => AMatName="M0", backend="cuda", mmtype="bstream-ksplit"
    // THIS IS ALWAYS THE TEMPLATE.
    meta.AMatName = baseName.substr(0, baseName.find('_'));
    meta.backend = baseName.substr(baseName.find('_') + 1, baseName.rfind('_') - baseName.find('_') - 1);
    meta.mmtype = baseName.substr(baseName.rfind('_') + 1);

    // Prefix meta.mmtype with gimmik_
    meta.mmtype = "gimmik_" + meta.mmtype;

    

    // meta.mmtype   = baseName.substr(underscorePos + 1);


    // Build .mtx path => e.g. "operators/p3/hex/M0.mtx"
    //     where meta.order = "3", meta.etype = "hex", meta.AMatName = "M0"
    std::string mtxPath = "operators/p" + meta.order + "/" + meta.etype
                          + "/" + meta.AMatName + ".mtx";
    return mtxPath;
}


int main(int argc, char* argv[])
{
    // -------------------------------------------------------------------------
    // 1. Parse command line
    //    Usage: <device> <kernelPath> <n> <niters>
    // -------------------------------------------------------------------------
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <device> <kernelPath> <n> <niters>\n";
        return 1;
    }

    // We'll manually create a CmdLineArgs struct just for consistency
    CmdLineArgs args;
    args.device     = argv[1];           // e.g. "H100"
    std::string kernelPath = argv[2];    // e.g. "kernels/gimmik/cuda/p3/hex/M0.cpp"
    args.n          = static_cast<size_t>(atoi(argv[3]));  // number of columns in B
    args.niters     = atoi(argv[4]);     // iteration count

    // -------------------------------------------------------------------------
    // 2. From the kernelPath, build the .mtx path => parse metadata
    // -------------------------------------------------------------------------
    FileMetadata meta;
    std::string mtxPath = kernelToMtxPath(kernelPath, meta);

    meta.device = args.device;

    // If mtxPath empty, warn and exit
    if (mtxPath.empty()) {
        cerr << "Error: unable to derive .mtx from kernel path: " << kernelPath << "\n";
        return 1;
    }

    // Read the matrix from .mtx => we only need (m, k) for sizing B, C
    // This function stores the matrix in col-major (m x k) in A_data
    vector<double> A_data;
    size_t m, k;
    readMTXdense(mtxPath, A_data, m, k, meta);

    // -------------------------------------------------------------------------
    // 3. Prepare B, C on host
    //    B => (k x n) all ones
    //    C => (m x n) all zeros
    // -------------------------------------------------------------------------
    vector<double> B_host(k * args.n, 1.0);
    vector<double> C_host(m * args.n, 0.0);

    // -------------------------------------------------------------------------
    // 4. Allocate device memory for B, C, copy data
    // -------------------------------------------------------------------------
    double *dB = nullptr, *dC = nullptr;
    cudaAllocDouble(&dB, B_host.size());
    cudaAllocDouble(&dC, C_host.size());

    cudaCopyToDevice(dB, B_host, B_host.size());
    cudaCopyToDevice(dC, C_host, C_host.size());

    // Note: Matrix A is baked in, Use A_data to verify correctness with cuBLAS.

    // -------------------------------------------------------------------------
    // 5. Create block/grid dimensions and do a warm-up kernel launch
    // -------------------------------------------------------------------------
    int blockSize = 128;  // could also use 256
    int gridSize  = (static_cast<int>(args.n) + blockSize - 1) / blockSize;

    // We pass n as the leading dimension in a row-major interpretation:
    //   B: shape (k x n), but the kernel expects B as [row i, col j].
    //   Treat B row-major => ldb = n. 
    //   Treat C row-major => ldc = n.
    int ldb = static_cast<int>(args.n);
    int ldc = static_cast<int>(args.n);

    gimmik_mm<<<gridSize, blockSize>>>(static_cast<int>(args.n), dB, ldb, dC, ldc);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // 6. Correctness check vs. cuBLAS (similar to dense/sparse)
    // -------------------------------------------------------------------------
    {
        // Create a cuBLAS handle
        cublasHandle_t blasHandle;
        cublasCreate(&blasHandle);

        // a) Allocate dA for the reference multiply
        double *dA = nullptr;
        cudaAllocDouble(&dA, A_data.size());
        cudaCopyToDevice(dA, A_data, A_data.size());

        // b) Allocate dR for the reference result => size (m x n)
        double *dR = nullptr;
        cudaAllocDouble(&dR, C_host.size());
        cudaMemset(dR, 0, C_host.size() * sizeof(double));

        // c) dR = A * B (col-major gemm)
        double alpha = 1.0, beta = 0.0;
        cublasDgemm(blasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(m),
                    static_cast<int>(args.n),
                    static_cast<int>(k),
                    &alpha,
                    dA, static_cast<int>(m),
                    dB, static_cast<int>(k),
                    &beta,
                    dR, static_cast<int>(m));
        cudaDeviceSynchronize();

        // d) Compare dC vs. dR using our helper
        bool ok = checkCublasGemmCorrectness(blasHandle,
                                             m, args.n, k,
                                             dA, dB, dC,
                                             alpha, beta,
                                             1e-6);
        if (!ok) {
            cerr << "[Warm-up] GiMMiK kernel results do not match reference.\n";
            // Clean up before returning
            cublasDestroy(blasHandle);
            cudaFreeDouble(dA);
            cudaFreeDouble(dR);
            return 1;
        }

        // e) Cleanup reference resources
        cublasDestroy(blasHandle);
        cudaFreeDouble(dA);
        cudaFreeDouble(dR);
    }

    // -------------------------------------------------------------------------
    // 7. Timed loop of the kernel launch
    // -------------------------------------------------------------------------
    auto start = high_resolution_clock::now();
    for (int i = 0; i < args.niters; i++) {
        gimmik_mm<<<gridSize, blockSize>>>(static_cast<int>(args.n), dB, ldb, dC, ldc);
    }
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();

    // Check for any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        // Still proceed to measure time, but it's indicative of an error.
    }

    double avg = duration<double>(end - start).count() / args.niters;

    // -------------------------------------------------------------------------
    // 8. Print CSV output (similar to dense/sparse)
    // -------------------------------------------------------------------------
    // We'll reuse the same "direct" backend label for consistent logging

    // Name this kernel launch as "gimmik_"+ name of the kernel.
    // For example, in kernels/gimmik/p3/hex/M0_cuda_bstream.cpp the name is bstream, which is the last part after underscore
    // We use the last part of the kernel path as the backend name.
    // e.g. "bstream"
    
    writeOutputCSV(meta, args.n, avg, efficiency(meta, m, k, args.n, avg));

    // -------------------------------------------------------------------------
    // 9. Cleanup
    // -------------------------------------------------------------------------
    cudaFreeDouble(dB);
    cudaFreeDouble(dC);

    return 0;
}
