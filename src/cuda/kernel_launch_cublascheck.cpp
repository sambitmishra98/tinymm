// Location: src/cuda/kernel_launch.cpp

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>


// Add these two lines if not already present:
#include <string>
#include <sstream>


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
 * \brief Parse the kernel path, e.g. 
 *        "kernels/gimmik/cuda/p1/hex/M0_bstream.cpp"
 *        now that your directory structure is:
 *          kernels/gimmik/<backend>/<pX>/<etype>/<AMatName>_<pattern>.cpp
 *
 * \param kernelPath (in) the path to the GiMMiK kernel .cpp file, 
 *        e.g. "kernels/gimmik/cuda/p1/hex/M0_bstream.cpp"
 * \param meta       (out) partially filled FileMetadata 
 *                   (order, etype, AMatName, backend, mmtype)
 * \return           the derived .mtx path, e.g. "operators/p1/hex/M0.mtx"
 */
std::string kernelToMtxPath(const std::string &kernelPath, FileMetadata &meta)
{
    // 1) Find the prefix "kernels/gimmik/" in kernelPath
    const std::string prefix = "kernels/gimmik/";
    size_t pos = kernelPath.find(prefix);
    if (pos == std::string::npos) {
        // If not found, we cannot parse it
        return "";
    }

    // Advance past "kernels/gimmik/"
    pos += prefix.size();
    // sub => e.g. "cuda/p1/hex/M0_bstream.cpp"
    std::string sub = kernelPath.substr(pos);

    // 2) Split 'sub' on '/' to get tokens:
    //    tokens[0] = "cuda"       (the backend)
    //    tokens[1] = "p1"         (polynomial order)
    //    tokens[2] = "hex"        (element type)
    //    tokens[3] = "M0_bstream.cpp"
    std::vector<std::string> tokens;
    {
        std::stringstream ss(sub);
        std::string part;
        while (std::getline(ss, part, '/'))
            tokens.push_back(part);
    }
    if (tokens.size() < 4) {
        // Must at least have [backend, pX, etype, filename]
        return "";
    }

    // 3) Parse the tokens into meta fields
    meta.backend = tokens[0];  // e.g. "cuda"

    // tokens[1] => e.g. "p1", skip 'p'
    // If tokens[1] = "p1", meta.order => "1"
    if (tokens[1].size() < 2 || tokens[1][0] != 'p') {
        // Something unexpected
        return "";
    }
    meta.order = tokens[1].substr(1);   // after 'p', e.g. "1"

    meta.etype = tokens[2];  // e.g. "hex"

    // Now parse filename => "M0_bstream.cpp" => remove ".cpp" => "M0_bstream"
    std::string fname = tokens[3];
    size_t dotPos = fname.rfind(".cpp");
    if (dotPos != std::string::npos) {
        fname = fname.substr(0, dotPos);
    }

    // Expect something like "M0_bstream" => split at underscore
    //   => M0  +  bstream
    size_t underscorePos = fname.find('_');
    if (underscorePos == std::string::npos) {
        return "";
    }
    meta.AMatName = fname.substr(0, underscorePos);       // e.g. "M0"
    std::string mmtypePart = fname.substr(underscorePos + 1); // e.g. "bstream"

    // 4) Set mmtype in meta: prefix "gimmik_"
    meta.mmtype = "gimmik_" + mmtypePart;

    // 5) Build the .mtx path => e.g. "operators/p1/hex/M0.mtx"
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

    // Note: The GiMMiK kernel presumably has matrix A baked in,
    //       so we do not need a device copy of A for the kernel itself.
    //       We only use A_data if we want to verify correctness with cuBLAS.

    // -------------------------------------------------------------------------
    // 5. Create block/grid dimensions and do a warm-up kernel launch
    // -------------------------------------------------------------------------
    int blockSize = 128;  // could also use 256
    int gridSize  = (static_cast<int>(args.n) + blockSize - 1) / blockSize;

    // We pass n as the leading dimension in a row-major interpretation:
    //   B has shape (k x n), but the kernel expects B as [row i, col j].
    //   We effectively treat B row-major => ldb = n. Same for C => ldc = n.
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
