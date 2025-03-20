#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

#include "src/base.h"       // parseCmdLineArgs, readMTXdense, etc.
#include "src/sycl/base.h"  // syclAllocDouble, syclCopyToDevice, etc.

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[])
{
    // 1) Parse command line
    CmdLineArgs args = parseCmdLineArgs(argc, argv);
    FileMetadata meta = parseFilename(args.matrixPath);
    meta.mmtype  = args.mmtype;  // e.g. "dense"
    meta.backend = "direct";
    meta.device  = args.device;

    // 2) Read A in row-major from .mtx
    vector<double> A_data;
    size_t m, k;
    // Our updated readMTXdense now stores A_data[row * k + col]
    readMTXdense(args.matrixPath, A_data, m, k, meta);

    // 3) Prepare B, C in row-major
    //    - B is shape (k × n) => each row has n columns => leading dimension = n
    //    - C is shape (m × n) => each row has n columns => leading dimension = n
    vector<double> B_data(k * args.n, 1.0);
    vector<double> C_data(m * args.n, 0.0);

    // 4) Create SYCL queue (row-major vs. col-major doesn't matter to the queue)
    sycl::queue q(sycl::gpu_selector_v);

    // 5) Allocate device memory
    double *dA = syclAllocDouble(q, A_data.size());
    double *dB = syclAllocDouble(q, B_data.size());
    double *dC = syclAllocDouble(q, C_data.size());

    // 6) Copy A, B, C to device
    syclCopyToDevice(q, dA, A_data);
    syclCopyToDevice(q, dB, B_data);
    syclCopyToDevice(q, dC, C_data);

    double alpha = 1.0;
    double beta  = 0.0;

    // 7) Warm-up with oneMKL row-major gemm
    //    A is (m × k) => ldA = k
    //    B is (k × n) => ldB = n
    //    C is (m × n) => ldC = n
    try {
        oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans, // A not transposed
            oneapi::mkl::transpose::nontrans, // B not transposed
            (int)m, (int)args.n, (int)k,
            alpha,
            dA, (int)k,  // leading dimension for A is k
            dB, (int)args.n, // leading dimension for B is n
            beta,
            dC, (int)args.n  // leading dimension for C is n
        );
        q.wait_and_throw();
    } catch (sycl::exception const &e) {
        cerr << "Warm-up GEMM failed: " << e.what() << "\n";
        return 1;
    }

    // 8) Check correctness vs. row-major reference gemm
    //    => This function must also call row_major::gemm with (ldA=k, ldB=n, ldC=n)
    bool ok = checkOneMKLGemmCorrectness(q, m, args.n, k, dA, dB, dC,
                                         alpha, beta, 1e-6);
    if (!ok) {
        cerr << "[Warm-Up] Results do not match reference.\n";
        return 1;
    }

    // 9) Timed loop
    auto start = high_resolution_clock::now();
    for(int i = 0; i < args.niters; i++) {
        oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            (int)m, (int)args.n, (int)k,
            alpha,
            dA, (int)k,
            dB, (int)args.n,
            beta,
            dC, (int)args.n
        );
    }
    q.wait();
    auto end = high_resolution_clock::now();

    double avg = duration<double>(end - start).count() / args.niters;

    // 10) CSV output
    writeOutputCSV(meta, args.n, avg, efficiency(meta, m, k, args.n, avg));

    // Cleanup
    syclFreeDouble(q, dA);
    syclFreeDouble(q, dB);
    syclFreeDouble(q, dC);

    return 0;
}
