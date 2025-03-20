
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "src/opencl/base.h"
#include <iostream>

// This matches the logic you used in the CUDA code but is now placed in OpenCL base.
std::string kernelToMtxPath(const std::string &kernelPath, FileMetadata &meta)
{
    // Example: kernelPath = "kernels/gimmik/p3/hex/M0_opencl_bstream.cpp"
    // 1) Find "kernels/gimmik/p"
    size_t pos = kernelPath.find("kernels/gimmik/p");
    if (pos == std::string::npos) {
        // Fallback if not found
        return "";
    }
    // Advance past "kernels/gimmik/"
    pos += std::string("kernels/gimmik/").size();

    // Next char => polynomial order, e.g. "3"
    // So "p3" => we skip 'p'
    pos++; // skip 'p'
    meta.order = kernelPath.substr(pos, 1);

    // Next slash => e.g. "...p3/hex..."
    size_t slashPos = kernelPath.find('/', pos);
    // Extract up to 3 letters for etype, e.g. "hex"
    meta.etype = kernelPath.substr(slashPos + 1, 3);

    // Final filename => e.g. "M0_opencl_bstream.cpp"
    std::string fname = kernelPath.substr(kernelPath.find_last_of('/') + 1);
    // Remove ".cpp"
    size_t dotPos = fname.rfind(".cpp");
    std::string baseName = (dotPos != std::string::npos)
                           ? fname.substr(0, dotPos)
                           : fname;

    // The filename portion presumably has the pattern "M0_opencl_bstream",
    // so we do the same logic as in the CUDA version:
    meta.AMatName = baseName.substr(0, baseName.find('_'));
    meta.backend  = baseName.substr(baseName.find('_') + 1,
                                    baseName.rfind('_') - baseName.find('_') - 1);
    meta.mmtype   = baseName.substr(baseName.rfind('_') + 1);

    // Prefix mmtype with "gimmik_"
    meta.mmtype = "gimmik_" + meta.mmtype;

    // Build .mtx path => e.g. "operators/p3/hex/M0.mtx"
    std::string mtxPath = "operators/p" + meta.order + "/" + meta.etype
                          + "/" + meta.AMatName + ".mtx";
    return mtxPath;
}
