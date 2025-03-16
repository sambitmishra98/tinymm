#include "src/common.h"  // contains parseFilename, readMTXMatrix, writeOutputCSV, etc.
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace std;
using namespace chrono;

// Minimal function to write a dense matrix in Matrix Market array format.
// Uses column-major order to match readMTXMatrix in "dense" mode.
static void writeMTXMatrix(const string &filename,
                           const vector<double> &matrix,
                           size_t rows, size_t cols)
{
    ofstream ofs(filename);
    if (!ofs) {
        cerr << "Cannot open output file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    ofs << "%%MatrixMarket matrix array real general\n";
    ofs << rows << " " << cols << "\n";

    // Because readMTXMatrix() stores in column-major:
    //   A_data[c * m + r] = val
    for (size_t c = 0; c < cols; c++) {
        for (size_t r = 0; r < rows; r++) {
            ofs << matrix[c * rows + r] << "\n";
        }
    }
    ofs.close();
}

// CPU-based matrix multiply: C = A × B
//   A is (m × k), B is (k × n), C is (m × n), all in column-major.
static void matMulCPU(const vector<double> &A,
                      const vector<double> &B,
                      vector<double> &C,
                      size_t m, size_t k, size_t n)
{
    // For each column c of B,
    //   for each row r of A,
    //      sum across the k dimension
    for (size_t c = 0; c < n; c++) {
        for (size_t r = 0; r < m; r++) {
            double sum = 0.0;
            for (size_t kk = 0; kk < k; kk++) {
                sum += A[kk * m + r] * B[c * k + kk];
            }
            C[c * m + r] = sum;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc < 8) {
        cerr << "Usage: " << argv[0]
             << " <A-mtx-file> <n> <iterations> <vendor> <device> <B-out-file> <C-out-file>\n";
        return 1;
    }

    // 1) Parse arguments
    string A_mtx = argv[1];
    size_t n = static_cast<size_t>(atoi(argv[2]));
    int iterations = atoi(argv[3]);
    string vendor = argv[4];
    string device = argv[5];
    string B_outfile = argv[6];
    string C_outfile = argv[7];

    // 2) Parse file metadata (order, etype, etc.)
    FileMetadata meta = parseFilename(A_mtx);

    // 3) Read A (m × k) from .mtx
    vector<double> A_data;
    size_t m, k;
    readMTXMatrix(A_mtx, A_data, m, k, meta);
    cout << "\nRead A: dimension " << m << " x " << k << " ("
         << (meta.sparsity == "sp" ? "sparse" : "dense") << ")\n";

    // 4) Create B (k × n) with a fixed random seed, or all ones if desired.
    //    Here we do random for demonstration.
    srand(1235);
    vector<double> B_data(k * n);
    for (size_t i = 0; i < k * n; i++) {
        B_data[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // 5) Store B in .mtx
    writeMTXMatrix(B_outfile, B_data, k, n);
    cout << "Wrote B: dimension " << k << " x " << n
         << " to " << B_outfile << endl;

    // 6) Prepare C (m × n) for the product
    vector<double> C_data(m * n, 0.0);

    // 7) Warm-up multiply once (optional)
    matMulCPU(A_data, B_data, C_data, m, k, n);

    // 8) Time multiple iterations of matMulCPU
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matMulCPU(A_data, B_data, C_data, m, k, n);
    }
    auto end = high_resolution_clock::now();
    double total_s = duration<double>(end - start).count();
    double avg_s = total_s / iterations;
    cout << "CPU Multiply time: " << total_s << "s over "
         << iterations << " iterations => " << avg_s << " s/iter\n";

    // 9) Store C in .mtx
    writeMTXMatrix(C_outfile, C_data, m, n);
    cout << "Wrote C: dimension " << m << " x " << n
         << " to " << C_outfile << endl;

    // 10) (Optional) Write CSV performance info
    //     We can reuse your existing function if you want it consistent.
    //     This calculates metrics using meta.nnz if the matrix was sparse, etc.
    writeOutputCSV(meta, n, m, k, iterations, avg_s, vendor, device);

    return 0;
}
