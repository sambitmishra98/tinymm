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
 */
std::string kernelToMtxPath(const std::string &kernelPath, FileMetadata &meta)
{
    // e.g. kernelPath = "kernels/gimmik/cuda/p3/hex/M0_bstream.cpp"
    // step 1: find "kernels/gimmik/p"
    size_t pos = kernelPath.find("kernels/gimmik/p");
    if (pos == std::string::npos) {
        // fallback
        return "";
    }
    pos += std::string("kernels/gimmik/").size() + 1; // skip "kernels/gimmik/"

    // next char => polynomial order, e.g. '3'
    meta.order = kernelPath.substr(pos, 1);

    // next slash => e.g. "...p3/hex..."
    size_t slashPos = kernelPath.find('/', pos);
    // read up to 3 or 4 letters for etype
    meta.etype = kernelPath.substr(slashPos + 1, 3);

    // final filename => e.g. "M0_bstream.cpp"
    std::string fname = kernelPath.substr(kernelPath.find_last_of('/') + 1);
    // remove ".cpp"
    size_t dotPos = fname.rfind(".cpp");
    std::string baseName = (dotPos != std::string::npos) 
                           ? fname.substr(0, dotPos)
                           : fname;

    // parse out AMatName vs mmtype from underscore
    // e.g. "M0_bstream-ksplit" => AMatName="M0", mmtype="bstream-ksplit"
    size_t underscorePos = baseName.find('_');
    meta.AMatName = baseName.substr(0, underscorePos);
    meta.mmtype   = baseName.substr(underscorePos + 1);

    // build .mtx path => e.g. "operators/p3/hex/M0.mtx"
    std::string mtxPath = "operators/p" + meta.order + "/" + meta.etype 
                          + "/" + meta.AMatName + ".mtx";
    return mtxPath;
}

int main(int argc, char* argv[])
{
    // 1. Parse command line (similar to dense/sparse)
    //    e.g. usage: <device> <kernelPath> <n> <niters>
    //    same shape as e.g.  ./gimmik_kernel_driver.exe H100 kernels/gimmik/... 1000 10
    //    We'll adapt parseCmdLineArgs for it:

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <device> <kernelPath> <n> <niters>\n";
        return 1;
    }

    // We'll manually create a CmdLineArgs struct:
    CmdLineArgs args;
    args.device     = argv[1];          // e.g. "H100"
    std::string kernelPath = argv[2];   // e.g. "kernels/gimmik/cuda/p3/hex/M0.cpp"
    args.n          = static_cast<size_t>(std::atoi(argv[3]));
    args.niters     = std::atoi(argv[4]);

    // 2. Parse kernel path => .mtx => parse file metadata
    FileMetadata meta;
    std::string mtxPath = kernelToMtxPath(kernelPath, meta);

    // read the matrix from .mtx => we only need (m, k) for sizing B, C
    std::vector<double> A_data;
    size_t m, k;
    readMTXdense(mtxPath, A_data, m, k, meta); // read as dense col-major
    // meta is updated w/ nnz if needed

    // 3. Prepare B, C on host
    //    B => (k x n), all ones
    //    C => (m x n), zero
    std::vector<double> B_host(k * args.n, 1.0);
    std::vector<double> C_host(m * args.n, 0.0);

    // 4. Allocate device memory for B, C
    double *dB = nullptr, *dC = nullptr;
    cudaAllocDouble(&dB, B_host.size());
    cudaAllocDouble(&dC, C_host.size());

    // copy B, C to device
    cudaCopyToDevice(dB, B_host, B_host.size());
    cudaCopyToDevice(dC, C_host, C_host.size());

    // The gimmik kernel presumably has matrix A baked in. 
    // So we do not need a device copy of A for the kernel itself.
    // But if we want to do a correctness check with cuBLAS, let's do so.

    // 5. Warm-up: call the GiMMiK kernel once
    int blockSize = 128;  // or 256
    int gridSize  = (args.n + blockSize - 1) / blockSize;

    // row-major interpretation => ldb = n, ldc = n
    int ldb = (int)args.n;
    int ldc = (int)args.n;

    gimmik_mm<<<gridSize, blockSize>>>( (int)args.n, dB, ldb, dC, ldc );
    cudaDeviceSynchronize();

    // 6. Correctness check (optional but consistent w/ your dense/sparse code):
    //    We'll do a reference multiply w/ cuBLAS => dR, compare to dC.
    {
        // a) allocate dA for reference. We read A_data in col-major (m x k).
        double *dA = nullptr;
        cudaAllocDouble(&dA, A_data.size());
        cudaCopyToDevice(dA, A_data, A_data.size());

        // b) allocate dR for reference result => size (m*n)
        double *dR = nullptr;
        cudaAllocDouble(&dR, C_host.size());
        cudaMemset(dR, 0, C_host.size() * sizeof(double));

        // c) create cuBLAS handle, do cublasDgemm => dR
        cublasHandle_t blasHandle;
        cublasCreate(&blasHandle);

        double alpha = 1.0, beta = 0.0;
        // cublas is col-major => A => (m x k), B => (k x n), C => (m x n)
        // leading dims => A => m, B => k, C => m
        cublasDgemm(blasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)m, (int)args.n, (int)k,
                    &alpha,
                    dA, (int)m,
                    dB, (int)k,
                    &beta,
                    dR, (int)m);
        cudaDeviceSynchronize();

        // d) copy dC (kernel result) and dR (reference) back to host
        std::vector<double> kernelC(m * args.n), refC(m * args.n);
        cudaCopyToHost(kernelC, dC, kernelC.size());
        cudaCopyToHost(refC,   dR, refC.size());

        // e) compare
        bool match = true;
        for (size_t i = 0; i < kernelC.size(); i++) {
            double diff = std::fabs(kernelC[i] - refC[i]);
            if (diff > 1e-6) {
                match = false;
                break;
            }
        }
        if (!match) { std::cerr << "Warm-up solution mismatch: Kernel result vs. cuBLAS reference.\n"; return 1; }

        // f) cleanup reference resources
        cublasDestroy(blasHandle);
        cudaFreeDouble(dA);
        cudaFreeDouble(dR);
    }

    // 7. Timed loop
    auto start = high_resolution_clock::now();
    for(int i = 0; i < args.niters; i++){
        gimmik_mm<<<gridSize, blockSize>>>( (int)args.n, dB, ldb, dC, ldc );
    }
    
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();

    // Wait for kernel to finish
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n"; }

    double avg = duration<double>(end - start).count() / args.niters;
    double bw  = processBandwidth(meta, m, k, args.n, avg);
    writeOutputCSV(args.device, meta, args.n, avg, bw);

    // 9. Cleanup
    cudaFreeDouble(dB);
    cudaFreeDouble(dC);

    return 0;
}
