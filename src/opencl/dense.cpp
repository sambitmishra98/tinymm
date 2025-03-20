// src/opencl/dense.cpp

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>

// Include tinytc’s C API header (make sure this is the correct one for your version)
#include "/home/sambit.mishra/.local/install/include/tinytc/tinytc_cl.h"  // or <tinytc.h> depending on your install

#include "src/base.h"       // For parseCmdLineArgs, readMTXdense, etc.
#include "src/opencl/base.h"  // For setupOpenCL(), teardownOpenCL(), checkTinyTCGemmCorrectness()

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[])
{
    // 1. Parse command-line and read matrix
    CmdLineArgs args = parseCmdLineArgs(argc, argv);
    FileMetadata meta = parseFilename(args.matrixPath);
    meta.mmtype = args.mmtype;

    vector<double> A_data;
    size_t m, k;
    readMTXdense(args.matrixPath, A_data, m, k, meta);

    // Prepare B (k x n, ones) and C (m x n, zeros)
    vector<double> B_data(k * args.n, 1.0);
    vector<double> C_data(m * args.n, 0.0);

    // 2. Setup OpenCL context, device, and queue
    cl_context context = nullptr;
    cl_device_id device = nullptr;
    cl_command_queue queue = nullptr;
    if (!setupOpenCL(context, device, queue)) {
        cerr << "Error: setupOpenCL() failed." << endl;
        return 1;
    }

    // 3. Create tinytc core info and source context
    tinytc_core_info_t info = nullptr;
    tinytc_status_t st = tinytc_cl_core_info_create(&info, device);
    if (st != 0 /*or your success code, e.g. TINYTc_SUCCESS*/) {
        cerr << "Error: tinytc_cl_core_info_create failed, st=" << (int)st << endl;
        return 1;
    }

    tinytc_source_context_t source_ctx = nullptr;
    st = tinytc_source_context(&source_ctx, NULL, 0);  // renamed function
    if (st != 0) {
        cerr << "Error: tinytc_create_source_context failed, st=" << (int)st << endl;
        return 1;
    }

    // 4. Allocate OpenCL buffers and copy A, B, C to device
    cl_int err = CL_SUCCESS;
    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_WRITE, A_data.size() * sizeof(double), nullptr, &err);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_WRITE, B_data.size() * sizeof(double), nullptr, &err);
    cl_mem dC = clCreateBuffer(context, CL_MEM_READ_WRITE, C_data.size() * sizeof(double), nullptr, &err);
    
    clEnqueueWriteBuffer(queue, dA, CL_TRUE, 0, A_data.size() * sizeof(double), A_data.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, dB, CL_TRUE, 0, B_data.size() * sizeof(double), B_data.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, dC, CL_TRUE, 0, C_data.size() * sizeof(double), C_data.data(), 0, nullptr, nullptr);

    // 5. Create the tall-and-skinny GEMM recipe (using the tinytc C API)
    double alpha = 1.0, beta = 0.0;
    int M = static_cast<int>(m), N = static_cast<int>(args.n), K = static_cast<int>(k);
    int ldA = M, ldB = K, ldC = M;
    
    tinytc_recipe_t recipe = nullptr;
    st = tinytc_create_recipe_ts_gemm(&recipe, info,
                                      M, N, K,
                                      ldA, ldB, ldC,
                                      alpha, beta,
                                      0, 0, 0,
                                      source_ctx);
    if (st != 0) {
        cerr << "Error: tinytc_create_recipe_ts_gemm failed, st=" << (int)st << endl;
        return 1;
    }

    // 6. Create a recipe handler
    tinytc_recipe_handler_t handler = nullptr;
    st = tinytc_recipe_handler(&handler, context, device, recipe, source_ctx);
    if (st != 0) {
        cerr << "Error: tinytc_create_recipe_handler failed, st=" << (int)st << endl;
        return 1;
    }

    // 7. Bind the device buffers to the recipe
    st = tinytc_set_recipe_ts_gemm_args(handler,
                                        (void*)dA, tinytc_mem_type_buffer,
                                        (void*)dB, tinytc_mem_type_buffer,
                                        (void*)dC, tinytc_mem_type_buffer);
    if (st != 0) {
        cerr << "Error: tinytc_set_recipe_ts_gemm_args failed, st=" << (int)st << endl;
        return 1;
    }

    // 8. Warm-up and correctness check
    st = tinytc_cl_recipe_handler_submit(handler, queue, 0, nullptr, nullptr);
    clFinish(queue);
    
    bool ok = checkTinyTCGemmCorrectness(context, queue, info, source_ctx, m, N, k,
                                         dA, dB, dC, alpha, beta, 1e-6);
    if (!ok) {
        cerr << "[Warm-Up] GEMM result mismatch." << endl;
        return 1;
    }

    // 9. Timed loop
    auto start_time = high_resolution_clock::now();
    for (int i = 0; i < args.niters; i++) {
        st = tinytc_cl_recipe_handler_submit(handler, queue, 0, nullptr, nullptr);
    }
    clFinish(queue);
    auto end_time = high_resolution_clock::now();
    double avg = duration<double>(end_time - start_time).count() / args.niters;

    writeOutputCSV(args.device, meta, args.n, avg, efficiency(meta, m, k, args.n, avg));

    // 10. Cleanup tinytc and OpenCL objects
    tinytc_recipe_handler(handler);
    tinytc_recipe(recipe);
    tinytc_core_info(info);
    tinytc_source_context(source_ctx);

    if(dA) clReleaseMemObject(dA);
    if(dB) clReleaseMemObject(dB);
    if(dC) clReleaseMemObject(dC);

    teardownOpenCL(context, queue);
    return 0;
}
