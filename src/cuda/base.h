#pragma once

#include <vector>
#include <cstddef>
#include <cublas_v2.h>

/**
 * Allocate device memory for a double array of length n.
 */
void cudaAllocDouble(double** dPtr, size_t n);

/**
 * Copy from host (std::vector<double>) to device array.
 */
void cudaCopyToDevice(double* dPtr, const std::vector<double>& hData, size_t n);

/**
 * Copy from device array to host (std::vector<double>).
 */
void cudaCopyToHost(std::vector<double>& hData, const double* dPtr, size_t n);

/**
 * Free a device double* pointer if non-null.
 */
void cudaFreeDouble(double* dPtr);

/**
 * \brief Perform a host-side correctness check on dC vs. a reference multiply in dR.
 *        That is, we do: R = alpha*A*B + beta*R (with cublas) on a temporary device array dR,
 *        then copy both dC and dR to host, compare them elementwise.
 *
 * \param handle  (in)   A valid cublas handle
 * \param m       (in)   #rows in A (and C)
 * \param n       (in)   #cols in B (and C)
 * \param k       (in)   #cols in A, #rows in B
 * \param dA      (in)   Device pointer to A (m*k)
 * \param dB      (in)   Device pointer to B (k*n)
 * \param dC      (in)   Device pointer to the "test" C  (m*n)
 * \param alpha   (in)   Alpha scalar for cublasDgemm
 * \param beta    (in)   Beta scalar for cublasDgemm
 * \param tol     (in)   Allowed numerical tolerance
 *
 * \return True if all elements match within tol, else false.
 *         If false, also prints the first few elements/differences.
 */
bool checkCublasGemmCorrectness(cublasHandle_t handle,
                                size_t m, size_t n, size_t k,
                                const double* dA,
                                const double* dB,
                                const double* dC,
                                double alpha,
                                double beta,
                                double tol = 1e-6);
