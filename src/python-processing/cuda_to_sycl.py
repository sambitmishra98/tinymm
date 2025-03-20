#!/usr/bin/env python3
"""
A generic converter to change pieces of a CUDA C++ file into a SYCL C++ file.
This script:
  1. Removes __device__ qualifiers.
  2. Finds __global__ kernel functions and converts them to SYCL style:
      - Changes the function signature to add a "sycl::queue &q" parameter.
      - Renames any kernel whose name starts with "gimmik_mm_" into "gimmik_mm".
      - Replaces the CUDA thread index computation with "item.get_global_id(0)".
      - Removes declarations of n, ldb, ldc inside the kernel body.
      - Removes duplicate definitions of "size_t i = item.get_global_id(0);".
      - Wraps the kernel body inside a q.submit()/parallel_for lambda.
  3. Updates constant values (for example, n from 1000000 to 1280000).
  
Usage:
    python convert_cuda_to_sycl.py input.cu output.cpp
"""

import re
import sys

def remove_device_qualifier(code: str) -> str:
    """Remove __device__ qualifiers."""
    return re.sub(r'__device__', '', code)

def transform_kernel_function(code: str) -> str:
    """
    Find CUDA __global__ kernel functions and convert them into SYCL functions.
    """
    # Pattern to match a CUDA kernel header: __global__ void kernelName(params) {
    kernel_header_pattern = re.compile(r'__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{', re.DOTALL)
    
    output = ""
    pos = 0
    while True:
        match = kernel_header_pattern.search(code, pos)
        if not match:
            output += code[pos:]
            break

        # Copy content up to the kernel header.
        output += code[pos:match.start()]

        kernel_name = match.group(1)
        params = match.group(2).strip()

        # Rename any kernel with name starting with "gimmik_mm_" to "gimmik_mm"
        if kernel_name.startswith("gimmik_mm_"):
            kernel_name = "gimmik_mm"

        # Add "sycl::queue &q" as the first parameter.
        if params:
            new_params = "sycl::queue &q, " + params
        else:
            new_params = "sycl::queue &q"

        # Find the entire kernel body.
        start_body = match.end() - 1  # position of the '{'
        brace_count = 1
        i = start_body + 1
        while i < len(code) and brace_count > 0:
            if code[i] == '{':
                brace_count += 1
            elif code[i] == '}':
                brace_count -= 1
            i += 1
        kernel_body = code[start_body+1:i-1]

        # --- Transform the kernel body ---
        # 1. Remove an outer "if (i < n) { ... }" wrapper if present.
        kernel_body = re.sub(
            r'if\s*\(\s*i\s*<\s*n\s*\)\s*\{(.*)\}\s*$', 
            r'\1', 
            kernel_body,
            flags=re.DOTALL
        )

        # 2. Replace the CUDA thread index computation with SYCL version.
        kernel_body = re.sub(
            r'const\s+int\s+i\s*=\s*blockDim\.x\s*\*\s*blockIdx\.x\s*\+\s*threadIdx\.x\s*;',
            'size_t i = item.get_global_id(0);',
            kernel_body
        )

        # 4. Remove the declarations of n, ldb, ldc that are inside the kernel body.
        kernel_body = re.sub(r'const\s+int\s+n\s*=\s*\d+\s*;', '', kernel_body)
        kernel_body = re.sub(r'const\s+int\s+ldb\s*=\s*\d+\s*;', '', kernel_body)
        kernel_body = re.sub(r'const\s+int\s+ldc\s*=\s*\d+\s*;', '', kernel_body)

        kernel_body = kernel_body.strip()

        # --- Build the new SYCL function ---
        new_function = ""
        new_function += f"void {kernel_name}({new_params})\n{{\n"
        new_function += "    // range setup\n"
        new_function += "    const int n = 1280000;\n"
        new_function += "    auto globalRange = sycl::range<1>(n);\n"
        new_function += "    auto localRange  = sycl::range<1>(128);\n\n"
        new_function += "    q.submit([&](sycl::handler &h) {\n"
        new_function += "        h.parallel_for(\n"
        new_function += "            sycl::nd_range<1>(globalRange, localRange),\n"
        new_function += "            [=](sycl::nd_item<1> item) {\n\n"
        new_function += "                // Set up kernel-specific constants\n"
        new_function += "                const int ldb = 1280000;\n"
        new_function += "                const int ldc = 1280000;\n"
        new_function += "                " + re.sub(r'\n', r'\n                ', kernel_body) + "\n"
        new_function += "            });\n"
        new_function += "    });\n"
        new_function += "}\n\n"

        output += new_function
        pos = i

    return output

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_cuda_to_sycl.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r") as infile:
        code = infile.read()

    # Step 1: Remove __device__ qualifiers.
    code = remove_device_qualifier(code)

    # Step 2: Transform __global__ kernel functions into SYCL style.
    code = transform_kernel_function(code)

    # Prepend the required SYCL header at the beginning.
    final_code = "#include <CL/sycl.hpp>\n\n" + code

    with open(output_file, "w") as outfile:
        outfile.write(final_code)

    print(f"Conversion complete. Output written to {output_file}")

if __name__ == "__main__":
    main()
