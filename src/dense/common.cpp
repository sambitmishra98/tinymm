#include "common.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

using namespace std;

// Revised parseFilename: use fixed relative locations
FileMetadata parseFilename(const string &mtx_file) 
{
    FileMetadata meta;

    size_t order_start = mtx_file.find("mats/"                 ) + 5;
    size_t order_end   = mtx_file.find('/'    , order_start + 1)    ;
    size_t etype_end   = mtx_file.find('/'    , order_end   + 1)    ;

    meta.order = mtx_file.substr(order_start + 1, order_end - order_start - 1);
    meta.etype = mtx_file.substr(order_end   + 1, etype_end - order_end   + 1);
    
    string filename = mtx_file.substr(etype_end + 1);
    size_t dash = filename.find('-'); meta.AMatName = filename.substr(0, dash);
    
    if      (filename.find("-sp") != string::npos) meta.sparsity = "sparse";
    else if (filename.find("-de") != string::npos) meta.sparsity = "dense";

    return meta;
}

void readMTXMatrix(const string &mtx_file, vector<double>& A_data, size_t &m, size_t &k) 
{
    ifstream infile(mtx_file);
    string line;
    
    // Header line, either  "%%MatrixMarket matrix coordinate real general"
    //                   or "%%MatrixMarket matrix array real general"
    getline(infile, line);
    bool isCoordinate = (line.find("coordinate") != string::npos);
    
    // Skip comment lines
    while (getline(infile, line)) {if (line[0] != '%') break; }
    
    istringstream iss(line);
    if (isCoordinate) 
    {
        int nnz;
        iss >> m >> k >> nnz;
        A_data.assign(m * k, 0.0);
        int i, j; double val; 
        for (int idx = 0; idx < nnz; idx++) {infile >> i >> j >> val; A_data[(j - 1) * m + (i - 1)] = val;}
    } 
    else 
    {
        iss >> m >> k;
        A_data.resize(m * k);
        for (size_t r = 0; r < m; r++) {for (size_t c = 0; c < k; c++) {double val; infile >> val; A_data[c * m + r] = val;}}
    }
    infile.close();
}

void writeOutputCSV(const FileMetadata &meta, size_t n, size_t m, size_t k, int iterations, double avg_sec, string &vendor, string &device) {

    string output_path = "results/" + vendor + "/" + device + "bench_dense.csv";
    ofstream output_file(output_path, ios::app);
    if (output_file.tellp() == 0) {output_file << "order,etype,opmat,sparsity,k,n,m,iterations,wtime\n";}
    output_file << meta.order << "," << meta.etype << "," << meta.AMatName << "," << meta.sparsity << "," << k << "," << n << "," << m << "," << iterations << "," << avg_sec << "\n";
    output_file.close();
}
