#include <CL/sycl.hpp>
#include "src/common.h"      // Your helper routines and FileMetadata
#include "kernels/manual/sycl/p3/hex/gimmik_mm.hpp" // Declaration for gimmik_mm(...)
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace sycl;

// Launch wrapper that does a warm-up and then times the kernel calls
double launch_gimmik_mm(queue &q, int m, int k, int n, const double *d_B, double *d_C, int iterations)
{
    // For a simple one-row kernel, we typically use these leading dims:
    int ldb = n;
    int ldc = n;

    // Warm-up launch
    gimmik_mm(q, n, d_B, ldb, d_C, ldc);
    q.wait(); // wait for warm-up to complete

    // Timed loop
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        gimmik_mm(q, n, d_B, ldb, d_C, ldc);
    }
    q.wait(); // Ensure all kernels have finished
    auto end = chrono::high_resolution_clock::now();

    double total = chrono::duration<double>(end - start).count();
    double avg   = total / iterations;

    cout << "\nTime taken: " << total << " s\n";
    return avg;
}

int main(int argc, char* argv[])
{
    // Parse input arguments
    if(argc < 6) {
        cerr << "Usage: " << argv[0] << " <m> <k> <n> <iterations> <device>\n";
        return 1;
    }
    // FileMetadata meta = parseFilename(argv[1]);
    size_t m          = atoi(argv[1]);   // for the A, C dimensions
    size_t k          = atoi(argv[2]);   // for the A, B dimensions
    size_t n          = atoi(argv[3]);   // for the B, C dimensions
    int iterations    = atoi(argv[4]);
    string device     = argv[5];

    // Read the matrix from .mtx to figure out (m, k). We do not actually use
    // A_data for the kernel, but we parse to fill metadata (like meta.nnz).
    // vector<double> A_data; 
    // size_t m, k;
    // readMTXMatrix(argv[1], A_data, m, k, meta);

    // Prepare host-side buffers for B (k x n) and C (m x n).
    vector<double> B_data(k*n, 1.0);  // fill with 1.0
    vector<double> C_data(m*n, 0.0);  // fill with 0.0

    // Create a SYCL queue targeting a GPU
    queue q{gpu_selector{}};

    // Allocate device memory with USM device allocations
    double *d_B = malloc_device<double>(B_data.size(), q);
    double *d_C = malloc_device<double>(C_data.size(), q);

    // Copy data from host to device
    q.memcpy(d_B, B_data.data(), B_data.size() * sizeof(double)).wait();
    q.memcpy(d_C, C_data.data(), C_data.size() * sizeof(double)).wait();

    // Launch and measure time
    double avg_sec = launch_gimmik_mm(q, m, k, n, d_B, d_C, iterations);

    // (Optional) Copy back results if you want to verify correctness
    // q.memcpy(C_data.data(), d_C, C_data.size() * sizeof(double)).wait();
    // for (size_t i = 0; i < m*n; i++) { ... }

    // Write CSV log
    // writeOutputCSV(meta, n, m, k, iterations, avg_sec, vendor, device);

    // Free device memory
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    return 0;
}
