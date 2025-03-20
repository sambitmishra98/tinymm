#pragma once

#include <string>
#include "src/base.h"  // For FileMetadata, etc.

/**
 * \brief Parse the kernel path, e.g. "kernels/gimmik/p3/hex/M0_opencl_bstream.cpp"
 *        to build a matching .mtx path "operators/p3/hex/M0.mtx".
 *        Similar logic to the CUDA version.
 */
std::string kernelToMtxPath(const std::string &kernelPath, FileMetadata &meta);

/**
 * Optionally, you could put more OpenCL utility prototypes here,
 * e.g. a checkOpenCLError(...) function, or device-querying functions, etc.
 */

