#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "common.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace sycl;
using namespace oneapi;

int main(int argc, char* argv[]) {
    FileMetadata meta = parseFilename(argv[1]);
    size_t n = (argc >= 3) ? atoi(argv[2]) : 0;
    int iterations = atoi(argv[3]);
    string vendor = argv[4];
    string device = argv[5];    

    vector<double> A_data;
    size_t m, k;
    readMTXMatrix(argv[1], A_data, m, k, meta);
    if(n == 0)
        n = k;
    
    vector<double> B_data(k * n, 1.0);
    vector<double> C_data(m * n, 0.0);
    
    // Create SYCL queue on a GPU device.
    queue q{gpu_selector_v};
    
    // Create SYCL buffers.
    buffer<double, 1> bufA(A_data.data(), range<1>(A_data.size()));
    buffer<double, 1> bufB(B_data.data(), range<1>(B_data.size()));
    buffer<double, 1> bufC(C_data.data(), range<1>(C_data.size()));
    
    double alpha = 1.0, beta = 0.0;
    
    try {
        //   d_C = alpha *   d_A  *  d_B  + beta *  d_C
        // m x n =          m x k * k x n +        m x n

        mkl::blas::gemm(q,
            mkl::transpose::nontrans, mkl::transpose::nontrans,
            m, n, k, 
            alpha, bufA, m,
            bufB, k,
            beta, bufC, m);
        q.wait_and_throw();
    } catch (const sycl::exception& e) {
        cerr << "GEMM warm-up exception: " << e.what() << "\n";
        return 1;
    }
    
    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        //   d_C = alpha *   d_A  *  d_B  + beta *  d_C
        // m x n =          m x k * k x n +        m x n
        mkl::blas::gemm(q,
            mkl::transpose::nontrans, mkl::transpose::nontrans,
            m, n, k, 
            alpha, bufA, m,
            bufB, k,
            beta, bufC, m);
    }
    q.wait_and_throw();
    auto end_time = chrono::high_resolution_clock::now();
    double total = chrono::duration<double>(end_time - start_time).count();
    double avg = total / iterations;
    
    writeOutputCSV(meta, n, m, k, iterations, avg, vendor, device);
    
    return 0;
}
