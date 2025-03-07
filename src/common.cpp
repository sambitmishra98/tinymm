#include "common.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
using namespace std;

FileMetadata parseFilename(const string &mtx_file) {
    FileMetadata meta;
    size_t pos = mtx_file.find("mats/p") + 6;
    meta.order = mtx_file.substr(pos, 1);                              // 1-digit order
    meta.etype = mtx_file.substr(mtx_file.find('/', pos) + 1, 3);        // 3-character etype
    string fname = mtx_file.substr(mtx_file.find_last_of('/') + 1);
    meta.AMatName = fname.substr(0, fname.find('-'));
    meta.sparsity = (fname.find("-sp") != string::npos) ? "sp" : "de";   // 2-character sparsity
    meta.nnz = 0;  // Initialize nnz
    return meta;
}

// Now readMTXMatrix sets meta.nnz as it reads the file.
void readMTXMatrix(const string &mtx_file, vector<double>& A_data, size_t &m, size_t &k, FileMetadata &meta) {
    ifstream infile(mtx_file);
    string line;
    while(getline(infile, line)) { if(line[0] != '%') break; }
    istringstream iss(line);
    if(meta.sparsity == "sp") {
        cout << "SPARSE" ;
        int nnz;
        iss >> m >> k >> nnz;
        meta.nnz = nnz;
        A_data.assign(m * k, 0.0);
        int i, j;
        double val;
        for (int idx = 0; idx < nnz; idx++) {
            infile >> i >> j >> val;
            A_data[(j - 1) * m + (i - 1)] = val;
        }
    } else {
        cout << "DENSE" ;

        iss >> m >> k;
        meta.nnz = m * k;
        A_data.resize(m * k);
        for (size_t r = 0; r < m; r++) {
            for (size_t c = 0; c < k; c++) {
                double val;
                infile >> val;
                A_data[c * m + r] = val;
            }
        }
    }
    infile.close();
}


// Helper function: Read a sparse Matrix Market file directly into CSR arrays.
// Assumes the file follows the "-sp.mtx" convention.
void readMTXCsrMatrix(const string &mtx_file, vector<int> &csrRowPtr, vector<int> &csrColInd, vector<double> &csrVal, size_t &m, size_t &k, FileMetadata &meta) 
{
    ifstream infile(mtx_file);
    if (!infile) {
    cerr << "Error opening file " << mtx_file << "\n";
    exit(EXIT_FAILURE);
    }
    // Skip comment lines (start with '%')
    string line;
    while (getline(infile, line)) {
    if (!line.empty() && line[0] != '%') break;
    }
    // The first non-comment line: m, k, nnz
    istringstream iss(line);
    int nnz;
    iss >> m >> k >> nnz;
    meta.nnz = nnz; // update metadata

    // Temporary storage for the coordinate (COO) entries.
    vector<int> rowIndices(nnz), colIndices(nnz);
    vector<double> values(nnz);
    for (int i = 0; i < nnz; i++) {
    int row, col;
    double val;
    infile >> row >> col >> val;
    // Convert from 1-indexed to 0-indexed.
    rowIndices[i] = row - 1;
    colIndices[i] = col - 1;
    values[i] = val;
    }
    infile.close();

    // Build CSR data structures.
    csrRowPtr.assign(m + 1, 0);
    // Count nonzeros per row.
    for (int i = 0; i < nnz; i++) {
    csrRowPtr[rowIndices[i] + 1]++;
    }
    // Exclusive prefix sum to form csrRowPtr.
    for (size_t i = 0; i < m; i++) {
    csrRowPtr[i + 1] += csrRowPtr[i];
    }
    csrColInd.resize(nnz);
    csrVal.resize(nnz);
    // Temporary copy of row pointers to use as insertion indices.
    vector<int> rowStart = csrRowPtr;
    for (int i = 0; i < nnz; i++) {
    int r = rowIndices[i];
    int pos = rowStart[r]++;
    csrColInd[pos] = colIndices[i];
    csrVal[pos] = values[i];
    }
}

// Compute performance metric: Cₘ = (2 * nnz * n) / avg_sec.
double processPerformance(const FileMetadata &meta, size_t n, double avg_sec) {
    return (2.0 * meta.nnz * n) / avg_sec;
}

// Compute effective bandwidth (BW) from reading B and writing C.
// For double precision, each element is 8 bytes.
// Here we assume B is of size (k x n) and C is of size (m x n).
double processBandwidth(size_t n, size_t m, size_t k, double avg_sec) {
    double total_bytes = 8.0 * (n * k + n * m);
    return total_bytes / avg_sec;
}

// writeOutputCSV now computes C_m using meta.nnz
// C_m = (2 * meta.nnz * n) / avg_sec, where n is the number of columns in B.
void writeOutputCSV(const FileMetadata &meta, size_t n, size_t m, size_t k, 
                    int iterations, double avg_sec, string &vendor, string &device) {

    string output_path = "results/" + vendor + "/" + device + "/bench_" + meta.sparsity + ".csv";
    ofstream output_file(output_path, ios::app);
    
    double Cm = processPerformance(meta, n, avg_sec);
    double BW = processBandwidth(n, m, k, avg_sec);
    double AI = Cm / BW;

    if (output_file.tellp() == 0)
//        output_file << "order,etype,opmat,sparsity,nnz,k,n,m,iterations,wtime,Cm,BW,AI\n";
    output_file << meta.order << "," << meta.etype << "," << meta.AMatName << "," << meta.sparsity << "," << k << "," << meta.nnz << "," << n << "," << m << "," << iterations << "," << avg_sec << "," << Cm << "," << BW << "," << AI << "\n";
    output_file.close();
}
