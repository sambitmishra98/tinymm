#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include "src/base.h"      // parseCmdLineArgs, readMTXdense, parseFilename, etc.
#include "src/sycl/base.h" // syclAllocDouble, syclCopyToDevice, etc.

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[])
{
    // --------------------------------------------------
    // 1) Parse command-line for device, mmtype, path, n, niters
    // --------------------------------------------------
    CmdLineArgs args = parseCmdLineArgs(argc, argv);

    FileMetadata meta = parseFilename(args.matrixPath);
    meta.mmtype  = args.mmtype; // e.g. "sparse"
    meta.backend = "direct";
    meta.device  = args.device;

    // --------------------------------------------------
    // 2) Read matrix A in row-major from .mtx
    //    (Make sure your readMTXdense has A_data[row*k + col] = val)
    // --------------------------------------------------
    vector<double> A_data;
    size_t m, k;
    readMTXdense(args.matrixPath, A_data, m, k, meta);
    int nnz = meta.nnz;

    // --------------------------------------------------
    // 3) Convert A (row-major) => CSR
    //    Each row has k columns => A_data[row*k + col]
    // --------------------------------------------------
    vector<int> rowPtr(m + 1, 0);
    vector<int> colInd(nnz);
    vector<double> vals(nnz);

    // pass 1: count nonzeros per row
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < k; col++) {
            double val = A_data[row * k + col];
            if (val != 0.0) {
                rowPtr[row + 1]++;
            }
        }
    }
    // prefix sum
    for (size_t i = 0; i < m; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }

    // pass 2: fill colInd and vals
    vector<int> rowStart = rowPtr;
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < k; col++) {
            double val = A_data[row * k + col];
            if (val != 0.0) {
                int dest = rowStart[row]++;
                colInd[dest] = (int)col;
                vals[dest]   = val;
            }
        }
    }

    // --------------------------------------------------
    // 4) Prepare B, C in row-major
    //    B => shape(k × n), leading dimension = n
    //    C => shape(m × n), leading dimension = n
    // --------------------------------------------------
    vector<double> B_host(k * args.n, 1.0);
    vector<double> C_host(m * args.n, 0.0);

    // --------------------------------------------------
    // 5) Create SYCL queue
    // --------------------------------------------------
    sycl::queue q(sycl::gpu_selector_v);

    // --------------------------------------------------
    // 6) Allocate device memory
    // --------------------------------------------------
    int*    dRowPtr = sycl::malloc_device<int>(rowPtr.size(), q);
    int*    dColInd = sycl::malloc_device<int>(colInd.size(), q);
    double* dVals   = syclAllocDouble(q, vals.size());
    double* dB      = syclAllocDouble(q, B_host.size());
    double* dC      = syclAllocDouble(q, C_host.size());

    // Copy
    q.memcpy(dRowPtr, rowPtr.data(), rowPtr.size() * sizeof(int));
    q.memcpy(dColInd, colInd.data(), colInd.size() * sizeof(int));
    q.memcpy(dVals, vals.data(), vals.size() * sizeof(double)).wait();

    syclCopyToDevice(q, dB, B_host);
    syclCopyToDevice(q, dC, C_host);

    // --------------------------------------------------
    // 7) Create an MKL sparse handle
    // --------------------------------------------------
    oneapi::mkl::sparse::matrix_handle_t A_handle;
    oneapi::mkl::sparse::init_matrix_handle(&A_handle);

    // --------------------------------------------------
    // 8) set_csr_data(...) using new signature:
    //    set_csr_data(queue, handle, nrows, ncols, index_base, rowPtr, colInd, vals)
    // --------------------------------------------------
    oneapi::mkl::sparse::set_csr_data(
        q,
        A_handle,
        (int64_t)m,
        (int64_t)k,
        oneapi::mkl::index_base::zero,
        dRowPtr,
        dColInd,
        dVals
    );

    // --------------------------------------------------
    // 9) Warm-up spMM => C, row-major
    //    The new gemm signature is:
    //    gemm(q, layout, opA, opB,
    //         alpha, A_handle,
    //         X, columns, ldx,
    //         beta,
    //         Y, ldy)
    //
    //    Since B is shape(k × n) row-major => columns=n, ldb=n
    //    C is shape(m × n) => ldc=n
    //    A is (m × k).
    // --------------------------------------------------
    double alpha = 1.0;
    double beta  = 0.0;
    try {
        oneapi::mkl::sparse::gemm(
            q,
            oneapi::mkl::layout::row_major,         // row-major
            oneapi::mkl::transpose::nontrans,       // A not trans
            oneapi::mkl::transpose::nontrans,       // B not trans
            alpha,
            A_handle,
            dB,                                     // X
            (int64_t)args.n,                        // columns in B => n
            (int64_t)args.n,                        // ldb => n
            beta,
            dC,                                     // Y
            (int64_t)args.n                         // ldc => n
        );
        q.wait_and_throw();
    } catch (sycl::exception const &e) {
        cerr << "Warm-up spMM failed: " << e.what() << "\n";
        return 1;
    }

    // --------------------------------------------------
    // 10) Check correctness vs. row-major dense gemm
    // --------------------------------------------------
    {
        // Copy A_data => dA for a reference gemm
        double* dA_dense = syclAllocDouble(q, A_data.size());
        q.memcpy(dA_dense, A_data.data(), A_data.size() * sizeof(double));

        // Also allocate dR => reference result
        double* dR = syclAllocDouble(q, C_host.size());
        q.memset(dR, 0, C_host.size() * sizeof(double));

        try {
            oneapi::mkl::blas::row_major::gemm(
                q,
                oneapi::mkl::transpose::nontrans, // A not trans
                oneapi::mkl::transpose::nontrans, // B not trans
                (int)m, (int)args.n, (int)k,
                alpha,
                dA_dense, (int)k,    // ldA = k (A is row-major m×k)
                dB, (int)args.n,     // ldB = n (B is row-major k×n)
                beta,
                dR, (int)args.n      // ldC = n (C is row-major m×n)
            );
            q.wait();
        } catch (sycl::exception const &e) {
            cerr << "Reference dense gemm failed: " << e.what() << "\n";
            return 1;
        }

        // Compare dC vs. dR
        bool ok = true;
        vector<double> hostC(m * args.n), hostR(m * args.n);
        q.memcpy(hostC.data(), dC, hostC.size() * sizeof(double)).wait();
        q.memcpy(hostR.data(), dR, hostR.size() * sizeof(double)).wait();

        for (size_t i = 0; i < hostC.size(); i++) {
            double diff = fabs(hostC[i] - hostR[i]);
            if (diff > 1e-6) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            cerr << "MISMATCH: spMM result != row-major dense reference.\n";
            return 1;
        }

        syclFreeDouble(q, dA_dense);
        syclFreeDouble(q, dR);
    }

    // --------------------------------------------------
    // 11) Timed loop
    // --------------------------------------------------
    auto start = high_resolution_clock::now();
    for(int i = 0; i < args.niters; i++) {
        oneapi::mkl::sparse::gemm(
            q,
            oneapi::mkl::layout::row_major,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            alpha,
            A_handle,
            dB,
            (int64_t)args.n,
            (int64_t)args.n,
            beta,
            dC,
            (int64_t)args.n
        );
    }
    q.wait();
    auto end = high_resolution_clock::now();
    double avg = duration<double>(end - start).count() / args.niters;

    // --------------------------------------------------
    // 12) CSV output
    // --------------------------------------------------
    writeOutputCSV(meta, args.n, avg, efficiency(meta, m, k, args.n, avg));

    // --------------------------------------------------
    // 13) Cleanup
    // --------------------------------------------------
    // New recommended usage of release_matrix_handle is:
    //  oneapi::mkl::sparse::release_matrix_handle(q, &A_handle);
    // but if you're okay with the old usage:
    oneapi::mkl::sparse::release_matrix_handle(q, &A_handle);

    syclFreeDouble(q, dB);
    syclFreeDouble(q, dC);
    syclFreeDouble(q, dVals);
    sycl::free(dRowPtr, q);
    sycl::free(dColInd, q);

    return 0;
}
