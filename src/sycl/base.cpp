#include "src/sycl/base.h"
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <iostream>
#include <cmath>   // for std::abs

//
// 1) Basic USM Memory Helpers
//

double* syclAllocDouble(sycl::queue &q, std::size_t n)
{
    // Allocate device memory using USM device allocations
    double* ptr = sycl::malloc_device<double>(n, q);
    return ptr;
}

void syclCopyToDevice(sycl::queue &q, double* dPtr, const std::vector<double>& hData)
{
    q.memcpy(dPtr, hData.data(), hData.size() * sizeof(double)).wait();
}

void syclCopyToHost(sycl::queue &q, std::vector<double>& hData, const double* dPtr)
{
    q.memcpy(hData.data(), dPtr, hData.size() * sizeof(double)).wait();
}

void syclFreeDouble(sycl::queue &q, double* dPtr)
{
    if (dPtr) {
        sycl::free(dPtr, q);
    }
}

//
// 2) OneMKL-based correctness check
//

bool checkOneMKLGemmCorrectness(sycl::queue &q,
                                std::size_t m,
                                std::size_t n,
                                std::size_t k,
                                const double* dA,
                                const double* dB,
                                const double* dC,
                                double alpha,
                                double beta,
                                double tol)
{
    // We'll allocate a temporary buffer dR on device for the reference result
    std::size_t lenC = m * n;
    double* dR = sycl::malloc_device<double>(lenC, q);

    // Zero initialize R
    q.memset(dR, 0, lenC * sizeof(double)).wait();

    // We wrap them in SYCL buffer-like calls, but using USM we can call oneMKL directly.
    // oneMKL supports USM versions of GEMM. We'll do a gemm call:
    try {
        oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,  // A not transposed
            oneapi::mkl::transpose::nontrans,  // B not transposed
            (int)m, (int)n, (int)k,
            alpha,
            dA, (int)k,    // leading dimension for row-major A is k
            dB, (int)n,    // leading dimension for row-major B is n
            beta,
            dR, (int)n     // leading dimension for row-major R is n
        );
        q.wait_and_throw();
    } catch (sycl::exception const &e) {
        std::cerr << "[checkOneMKLGemmCorrectness] oneMKL gemm exception: " << e.what() << "\n";
        sycl::free(dR, q);
        return false;
    }

    // Copy both dC and dR back to host
    std::vector<double> C_host(lenC), R_host(lenC);
    q.memcpy(C_host.data(), dC, lenC * sizeof(double)).wait();
    q.memcpy(R_host.data(), dR, lenC * sizeof(double)).wait();

    // Compare element-wise
    bool match = true;
    for (std::size_t i = 0; i < lenC; i++) {
        double diff = std::abs(R_host[i] - C_host[i]);
        if (diff > tol) {
            match = false;
            break;
        }
    }

    if (!match) {
        std::cerr << "[checkOneMKLGemmCorrectness] Mismatch found!\n";
        std::cerr << "First 10 elements from test (C) vs. reference (R):\n";
        for (std::size_t i = 0; i < 10 && i < lenC; i++) {
            double diff = std::abs(R_host[i] - C_host[i]);
            std::cerr << "  idx " << i
                      << ": C=" << C_host[i]
                      << ", R=" << R_host[i]
                      << ", diff=" << diff << "\n";
        }
    }

    sycl::free(dR, q);
    return match;
}
