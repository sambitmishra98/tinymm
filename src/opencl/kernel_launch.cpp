// Location: src/opencl/kernel_launch.cpp

// Define OpenCL 300
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "src/opencl/base.h"   // kernelToMtxPath, etc.
#include "src/base.h"          // readMTXdense, FileMetadata, etc.
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

/**
 * \brief Read an OpenCL kernel source file and return its contents as a string.
 */
static std::string readKernelSource(const std::string &kernelPath)
{
    std::ifstream ifs(kernelPath);
    if (!ifs.is_open()) {
        cerr << "Error: cannot open OpenCL kernel file: " << kernelPath << "\n";
        return "";
    }
    std::string source((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    return source;
}

int main(int argc, char* argv[])
{
    //--------------------------------------------------------------------------
    // 1) Parse Command Line: usage <device> <kernelPath> <n> <niters>
    //--------------------------------------------------------------------------
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <device> <kernelPath> <n> <niters>\n";
        return 1;
    }

    CmdLineArgs args;
    args.device     = argv[1];          // e.g. "H100" or "max1550"
    std::string kernelPath = argv[2];   // e.g. "kernels/gimmik/p3/hex/M0_opencl_bstream.cl"
    args.n          = static_cast<size_t>(atoi(argv[3]));
    args.niters     = atoi(argv[4]);

    //--------------------------------------------------------------------------
    // 2) From kernelPath => parse metadata, get .mtx path
    //--------------------------------------------------------------------------
    FileMetadata meta;
    std::string mtxPath = kernelToMtxPath(kernelPath, meta);
    meta.device = args.device;
    if (mtxPath.empty()) {
        cerr << "Error: unable to derive .mtx from kernel path: " << kernelPath << "\n";
        return 1;
    }

    // Read matrix from .mtx
    vector<double> A_data;
    size_t m, k;
    readMTXdense(mtxPath, A_data, m, k, meta);

    // Prepare host B, C
    vector<double> B_host(k * args.n, 1.0);
    vector<double> C_host(m * args.n, 0.0);

    //--------------------------------------------------------------------------
    // 3) OpenCL initialization
    //--------------------------------------------------------------------------
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        cerr << "No OpenCL platforms found.\n";
        return 1;
    }
    // For simplicity, pick the first platform
    cl::Platform platform = platforms[0];

    // Get devices from that platform
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        cerr << "No OpenCL GPU devices found.\n";
        return 1;
    }
    cl::Device device = devices[0];

    // Create context and command queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    //--------------------------------------------------------------------------
    // 4) Create OpenCL buffers
    //--------------------------------------------------------------------------
    size_t sizeB = B_host.size() * sizeof(double);
    size_t sizeC = C_host.size() * sizeof(double);

    // B = read-only, C = read-write
    cl::Buffer dB(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeB, B_host.data());
    cl::Buffer dC(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeC, C_host.data());

    //--------------------------------------------------------------------------
    // 5) Build OpenCL program from kernel source
    //--------------------------------------------------------------------------
    // We assume kernelPath points to a .cl file, e.g. "kernels/gimmik/p3/hex/M0_opencl_bstream.cl"
    std::string kernelSource = readKernelSource(kernelPath);
    if (kernelSource.empty()) {
        cerr << "Error: kernel source from " << kernelPath << " is empty.\n";
        return 1;
    }

    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.size()});
    cl::Program program(context, sources);

    // Build with chosen compiler options
    cl_int buildErr = program.build({device}, "-cl-std=CL1.2");
    if (buildErr != CL_SUCCESS) {
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        cerr << "OpenCL build error: " << buildErr << "\n";
        cerr << "Build log:\n" << buildLog << "\n";
        return 1;
    }

    // Create kernel (symbol name from the .cl file)
    cl::Kernel gimmikKernel(program, "gimmik_mm");

    //--------------------------------------------------------------------------
    // 6) Set kernel arguments
    //--------------------------------------------------------------------------
    int n_int   = static_cast<int>(args.n);
    int ldb_int = static_cast<int>(args.n);  // row-major => ldb = n
    int ldc_int = static_cast<int>(args.n);  // row-major => ldc = n

    gimmikKernel.setArg(0, n_int);
    gimmikKernel.setArg(1, dB);
    gimmikKernel.setArg(2, ldb_int);
    gimmikKernel.setArg(3, dC);
    gimmikKernel.setArg(4, ldc_int);

    // The kernel might have __attribute__((reqd_work_group_size(64, 2, 1))) => local=(64,2)
    // Adjust global sizes so dimension 0 covers 'n'
    size_t localX = 64;
    size_t localY = 2;
    size_t globalX = ((args.n + localX - 1) / localX) * localX;
    size_t globalY = 2;

    //--------------------------------------------------------------------------
    // 7) Warm-up kernel launch
    //--------------------------------------------------------------------------
    queue.enqueueNDRangeKernel(
        gimmikKernel,
        cl::NullRange,
        cl::NDRange(globalX, globalY),
        cl::NDRange(localX,  localY)
    );
    queue.finish();

    //--------------------------------------------------------------------------
    // 8) (No correctness check - purely OpenCL)
    //--------------------------------------------------------------------------
    // Skipped any CUDA/cublas checking.

    //--------------------------------------------------------------------------
    // 9) Timed loop
    //--------------------------------------------------------------------------
    auto start = high_resolution_clock::now();
    for (int i = 0; i < args.niters; i++) {
        queue.enqueueNDRangeKernel(
            gimmikKernel,
            cl::NullRange,
            cl::NDRange(globalX, globalY),
            cl::NDRange(localX, localY)
        );
    }
    queue.finish();
    auto end = high_resolution_clock::now();

    double avg_sec = duration<double>(end - start).count() / args.niters;

    //--------------------------------------------------------------------------
    // 10) Print CSV output
    //--------------------------------------------------------------------------
    double eff = efficiency(meta, m, k, args.n, avg_sec);
    writeOutputCSV(meta, args.n, avg_sec, eff);

    //--------------------------------------------------------------------------
    // 11) Done
    //--------------------------------------------------------------------------
    return 0;
}
