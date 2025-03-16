import numpy as np
from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from scipy.io import mmread

import os
import sys


def benchmark(backend, a_np, N, beta):
    M, K = a_np.shape
    dsize = np.dtype(backend.fpdtype).itemsize

    b_np = np.random.randn(K, N)
    c_np = np.random.randn(M, N)

    a_be = backend.const_matrix(a_np)
    b_be = backend.matrix(b_np.shape, b_np, tags={'align'})
    c_be = backend.matrix(c_np.shape, c_np, tags={'align'})

    kern = backend.kernel('mul', a_be, b_be, c_be, beta=beta)

    fp_dense = 2*M*N*K / kern.dt / 1024**3
    fp_sparse = 2*N*np.count_nonzero(a_np) / kern.dt / 1024**3
    bw = (M + (M if beta else 0) + K)*N*dsize / kern.dt / 1024**3

    return fp_dense, fp_sparse, bw


def main():
    BNAME = 'cuda'
    N = 10
    PREC = 'double'
    ELES = ['hex', 'pri']
    ORDERS = [3]
    MATS = ['m0']

    inistr = f'''
    [backend]
    precision = {PREC}
    '''
    ini = Inifile(inistr)
    backend = get_backend(BNAME, ini)

    for p in ORDERS:
        for e in ELES:
            for m in MATS:
                path = '/home/sambit.mishra/scratch/03_KERNELPERFORMANCE/tinymm-benchmarking/operators/p3/hex/M0.mtx'
                a_np = mmread(path).todense()
                beta = 1 if m in ('m3', 'm6') else 0

                fp_dense, fp_sparse, bw = benchmark(backend, a_np, N, beta)

                print(p, e, m, fp_dense, fp_sparse, bw)


if __name__ == '__main__':
    main()