#pragma once

#include <vector>
#include <cstddef>
#include <CL/sycl.hpp>
#include "src/base.h" // For FileMetadata, etc.

/**
 * \brief Allocate device memory for a double array of length n using SYCL USM.
 */
double* syclAllocDouble(sycl::queue &q, std::size_t n);

/**
 * \brief Copy from host (std::vector<double>) to device array (USM pointer).
 */
void syclCopyToDevice(sycl::queue &q, double* dPtr, const std::vector<double>& hData);

/**
 * \brief Copy from device array (USM pointer) to host (std::vector<double>).
 */
void syclCopyToHost(sycl::queue &q, std::vector<double>& hData, const double* dPtr);

/**
 * \brief Free a device USM pointer (double*).
 */
void syclFreeDouble(sycl::queue &q, double* dPtr);

/**
 * \brief Perform a host-side correctness check on dC vs. a reference multiply in dR 
 *        using oneMKL. This replicates the logic from the CUDA version’s
 *        cublas-based check but uses SYCL/oneMKL for the reference multiply.
 *
 * \param q       (in)   SYCL queue
 * \param m, n, k (in)   Matrix dimensions
 * \param dA      (in)   Device pointer to A (m*k)
 * \param dB      (in)   Device pointer to B (k*n)
 * \param dC      (in)   Device pointer to the "test" C  (m*n)
 * \param alpha   (in)   Scalar alpha
 * \param beta    (in)   Scalar beta
 * \param tol     (in)   Allowed numerical tolerance
 *
 * \return True if all elements match within tol, else false. 
 *         If false, also prints the first few differences.
 */
bool checkOneMKLGemmCorrectness(sycl::queue &q,
                                std::size_t m,
                                std::size_t n,
                                std::size_t k,
                                const double* dA,
                                const double* dB,
                                const double* dC,
                                double alpha,
                                double beta,
                                double tol = 1e-6);
