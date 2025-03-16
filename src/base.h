#pragma once

#include <string>
#include <vector>
#include <cstddef>  // for size_t

// ----------------------------------------------------------
// 1) Basic structs
// ----------------------------------------------------------

/// Metadata about the matrix file (order, etype, etc.).
struct FileMetadata {
    std::string mmtype;   // "dense" / "sparse" / "kernel" / ...
    std::string order;    // e.g. "3"
    std::string etype;    // e.g. "hex"
    std::string AMatName; // e.g. "M0"
    int nnz = 0;          // #nonzeros (updated when reading .mtx)
};

/// Encapsulate command line arguments in a single struct.
struct CmdLineArgs {
    std::string device;   // e.g. "H100/A100/..."
    std::string mmtype;   // e.g. "dense" / "sparse"
    std::string matrixPath; // e.g. "operators/p3/hex/M0.mtx"
    size_t n;             // columns of B (and C)
    int niters;           // iteration count for timed loop
};

// ----------------------------------------------------------
// 2) Command-line + file parsing
// ----------------------------------------------------------

/**
 * \brief Parse the standard 5-argument command line:
 *        <device> <mmtype> <matrix.mtx> <n> <niters>.
 */
CmdLineArgs parseCmdLineArgs(int argc, char* argv[]);

/**
 * \brief Parse the matrix file path (e.g., "operators/p3/hex/M0.mtx")
 *        to fill FileMetadata (order, etype, AMatName).
 */
FileMetadata parseFilename(const std::string &mtx_file);

// ----------------------------------------------------------
// 3) Matrix reading
// ----------------------------------------------------------

/**
 * \brief Read a .mtx file and store result into a dense array A_data.
 *        The matrix is stored in column-major layout for use with cublasDgemm.
 *        If the file is a sparse .mtx, this function still “densifies” it.
 */
void readMTXdense(const std::string &mtx_file,
                  std::vector<double> &A_data,
                  size_t &m, size_t &k,
                  FileMetadata &meta);

// ----------------------------------------------------------
// 4) Bandwidth & CSV output
// ----------------------------------------------------------

/**
 * \brief Compute an approximate memory bandwidth metric:
 *        If mmtype == "dense", use (16 * m*k*n) / time
 *        If mmtype == "sparse", use (16 * nnz * n) / time
 */
double processBandwidth(const FileMetadata &meta,
                        size_t m, size_t k, size_t n,
                        double avg_sec);

/**
 * \brief Append a line to results/benchmarks.csv with device, mmtype, dims, etc.
 */
void writeOutputCSV(const std::string &device,
                    const FileMetadata &meta,
                    size_t n,
                    double avg,
                    double bw);

// ----------------------------------------------------------
// 5) Write a dense matrix C to a .mtx file for debugging
// ----------------------------------------------------------
/**
 * \brief Write the dense matrix C (m x n) to disk in .mtx (COO) format.
 *        This is purely for offline inspection.
 */
void writeDenseMatrixToMTX(const std::string &outFile,
                           const std::vector<double> &C,
                           size_t m, size_t n);
