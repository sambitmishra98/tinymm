#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <cstddef>

// Parsed MTX file name, matrices always in "samples/pyfr/mats"
// ../../samples/pyfr/mats/p<order>/<mesh_type>-<OpMat>-<dims>-<sp/de>.mtx
struct FileMetadata 
{
    std::string order;
    std::string etype;
    std::string AMatName;
    std::string sparsity;
};

FileMetadata parseFilename(const std::string &filename);
void readMTXMatrix(const std::string &mtx_file, std::vector<double>& A_data, std::size_t &m, std::size_t &k);
void writeOutputCSV(const FileMetadata &meta, std::size_t n, std::size_t m, std::size_t k, int iterations, double avg_sec, std::string &vendor, std::string &device);

#endif // COMMON_H
