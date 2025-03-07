#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
using namespace std;

struct FileMetadata {
    string order;
    string etype;
    string AMatName;
    string sparsity;
    int nnz;
};

FileMetadata parseFilename(const string &mtx_file);
void readMTXMatrix(const string &mtx_file, vector<double>& A_data, size_t &m, size_t &k, FileMetadata &meta);
void readMTXCsrMatrix(const string &mtx_file, vector<int> &csrRowPtr, vector<int> &csrColInd, vector<double> &csrVal, size_t &m, size_t &k, FileMetadata &meta);
void writeOutputCSV(const FileMetadata &meta, size_t n, size_t m, size_t k, int iterations, double avg_sec, string &vendor, string &device);

#endif // COMMON_H
