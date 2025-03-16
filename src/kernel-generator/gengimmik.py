#!/usr/bin/env python3

# Location: src/kernel-generator/gengimmik.py

import os
import argparse
import numpy as np
from scipy.io import mmread

from gimmik.c import CMatMul
from gimmik.copenmp import COpenMPMatMul
from gimmik.cuda import CUDAMatMul
from gimmik.ispc import ISPCMatMul
from gimmik.hip import HIPMatMul
from gimmik.metal import MetalMatMul
from gimmik.opencl import OpenCLMatMul


def kernel_generator(mat, dtype, platform, alpha=1.0, beta=0.0, funcn='gimmik_mm',
                n=None, ldb=None, ldc=None):
    import warnings

    platmap = {
        'c': CMatMul,
        'c-omp': COpenMPMatMul,
        'cuda': CUDAMatMul,
        'ispc': ISPCMatMul,
        'hip': HIPMatMul,
        'opencl': OpenCLMatMul
    }

    mm = platmap[platform](alpha*mat, beta, None, n, ldb, ldc)
    return mm.kernels(dtype, kname=funcn)

def main():
    parser = argparse.ArgumentParser(description='Generate GiMMiK kernels, given operator matrix as *.mtx.')
    parser.add_argument('order', type=int, help='Polynomial order', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],         default=3)
    parser.add_argument('etype', type=str, help='Element type'    , choices=['hex'],                             default='hex')
    parser.add_argument('mat'  , type=str, help='Matrix name'     , choices=['M0', 'M3', 'M6', 'M132', 'M460'],  default='M0')
    parser.add_argument('be'   , type=str, help='Backend'         , choices=['cuda', 'hip', 'opencl', 'openmp'], default='cuda')

    args = parser.parse_args()
    matrix_path = os.path.join("operators/", f"p{args.order}", args.etype, f"{args.mat}.mtx")
    sparse_mat = mmread(matrix_path)
    dense_mat = sparse_mat.toarray()

    kgen = kernel_generator(dense_mat, dtype=np.double, platform=args.be)

    kern = 0
    while kern < 16:
        try:
            kernel = next(kgen)
            code = kernel[0]
            kname = kernel[1]['tplname']
        except StopIteration:
            print(f"Only {kern} kernels generated. Stopping.")
            break
        kern += 1

        # Write kernel to results/kernels/gimmik/
        out_dir = os.path.join("kernels/gimmik", f"p{args.order}", args.etype)
        os.makedirs(out_dir, exist_ok=True)

        out_fname = f"{args.mat}_{args.be}_{kname}.cpp"
        out_path = os.path.join(out_dir, out_fname)

        with open(out_path, 'w') as f:
            f.write(code)

        print(f"'{args.be}' backend kernel generated and written to {out_path}")

if __name__ == "__main__":
    main()
