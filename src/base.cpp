// Location: "src/base.cpp"

#include "src/base.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>

// -------------------------------------------------------------------
// 1) Parse command line
// -------------------------------------------------------------------
CmdLineArgs parseCmdLineArgs(int argc, char* argv[])
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <device> <mmtype> <matrix.mtx> <n> <niters>\n";
        std::exit(1);
    }
    CmdLineArgs args;
    args.device     = argv[1];           // e.g. "H100"
    args.mmtype     = argv[2];           // e.g. "dense"
    args.matrixPath = argv[3];           // e.g. "operators/p3/hex/M0.mtx"
    args.n          = static_cast<size_t>(std::atoi(argv[4]));
    args.niters     = std::atoi(argv[5]);

    return args;
}

// -------------------------------------------------------------------
// 2) Parse filename -> FileMetadata
// -------------------------------------------------------------------
FileMetadata parseFilename(const std::string &mtx_file)
{
    // Typical format: "operators/p<order>/<etype>/<AMatName>.mtx"
    FileMetadata meta;

    // find "operators/p"
    size_t pos = mtx_file.find("operators/p");
    if (pos == std::string::npos) {
        // fallback or error
        meta.order = "?";
        meta.etype = "?";
        meta.AMatName = mtx_file;
        return meta;
    }
    // skip "operators/p"
    pos += 11;
    // next char is polynomial order
    meta.order = mtx_file.substr(pos, 1);

    // read up to next slash for etype
    size_t slashPos = mtx_file.find('/', pos);
    if (slashPos == std::string::npos) {
        meta.etype = "?";
        meta.AMatName = mtx_file;
        return meta;
    }
    // e.g. "hex"
    meta.etype = mtx_file.substr(slashPos + 1, 4);
    // remove trailing '/' if any
    if (!meta.etype.empty() && meta.etype.back() == '/')
        meta.etype.pop_back();

    // final part: e.g. "M0.mtx"
    std::string fname = mtx_file.substr(mtx_file.find_last_of('/') + 1);
    // remove ".mtx"
    size_t dotPos = fname.find(".mtx");
    if (dotPos != std::string::npos)
        meta.AMatName = fname.substr(0, dotPos);
    else
        meta.AMatName = fname;

    meta.nnz = 0;
    return meta;
}

// -------------------------------------------------------------------
// 3) readMTXdense
// -------------------------------------------------------------------
void readMTXdense(const std::string &mtx_file,
                  std::vector<double> &A_data,
                  size_t &m, size_t &k,
                  FileMetadata &meta)
{
    std::ifstream infile(mtx_file);
    if (!infile) {
        std::cerr << "Error: cannot open " << mtx_file << "\n";
        std::exit(1);
    }

    // skip comment lines (start with '%')
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    // parse "m k nnz"
    int mm, kk, nnz;
    {
        std::istringstream iss(line);
        iss >> mm >> kk >> nnz; // typical .mtx line for dimension
    }
    m = static_cast<size_t>(mm);
    k = static_cast<size_t>(kk);

    meta.nnz       = nnz;
    meta.opmatsize = m * k;

    // allocate dense array of size m*k, fill with 0
    A_data.assign(m * k, 0.0);

    // read the (row, col, val) lines
    for (int i = 0; i < nnz; i++) {
        int row, col;
        double val;
        infile >> row >> col >> val;
        // matrix uses 1-based indexing => convert to 0-based
        // store in column-major => A[col * m + row]
        A_data[(col - 1) * m + (row - 1)] = val;
    }
    infile.close();
}

double efficiency(const FileMetadata &meta,
                        size_t m, size_t k, size_t n,
                        double avg_sec)
{
    double bw = 2e12; // 2 TB/s

    return 8 * n * (m + k) /bw/ avg_sec;
}

void writeOutputCSV(const std::string &device,
                    const FileMetadata &meta,
                    size_t n,
                    double avg,
                    double eff)
{
    std::string outFile = "results/benchmarks.csv";
    std::ofstream out(outFile, std::ios::app);
    if (!out) {
        std::cerr << "Error: cannot open " << outFile << "\n";
        return;
    }

    // write header if empty
    if (out.tellp() == 0) {
        out << "device,mmtype,order,etype,AMatName,AMatSize,AMatnnz,n,wtime,efficiency\n";
    }

    out << device << ","
        << meta.mmtype << ","
        << meta.order << ","
        << meta.etype << ","
        << meta.AMatName << ","
        << meta.opmatsize << ","
        << meta.nnz << ","
        << n << ","
        << avg << ","
        << eff << "\n";

    out.close();
}

// -------------------------------------------------------------------
// 6) Write a dense matrix C to .mtx
//    (This can help you manually verify correctness offline.)
// -------------------------------------------------------------------
void writeDenseMatrixToMTX(const std::string &outFile,
                           const std::vector<double> &C,
                           size_t m, size_t n)
{
    // For a fully dense m x n, the #nonzeros = m*n
    std::ofstream ofs(outFile);
    if (!ofs) {
        std::cerr << "Error: cannot open " << outFile << " for writing.\n";
        return;
    }

    long long nnz = (long long)m * (long long)n;

    // Write the matrix market style header:  "m n nnz"
    ofs << m << " " << n << " " << nnz << "\n";

    // Now output each entry in row-major order:
    // row col value   (1-based indexing)
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            double val = C[row * n + col];
            ofs << (row + 1) << " " << (col + 1) << " " << val << "\n";
        }
    }
    ofs.close();
}
