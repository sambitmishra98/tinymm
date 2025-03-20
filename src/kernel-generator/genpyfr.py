#!/usr/bin/env python3

# Location: src/kernel-generator/genpyfr.py

import os
import sys
import numpy as np
from scipy.io import mmread

# PyFR imports
from pyfr.backends import get_backend
from pyfr.inifile import Inifile

def parse_mtx_filename(mtx_file):
    parts = mtx_file.strip().split('/')
    order = parts[1][1:] if len(parts := mtx_file.split('/')) > 2 else '?'
    etype = parts[2] if len(parts := mtx_file.split('/')) > 2 else '?'
    amat = os.path.basename(mtx_file).replace('.mtx', '')
    return order, etype, amat

def ideal_performance(m, k, n):
    bw = 2e12
    return (8*n*(m+k)/bw)

def compute_efficiency(m, k, n, avg_sec):
    return ideal_performance(m, n, k)/avg_sec

def main():
    if len(sys.argv) < 7:
        print(f"Usage: {sys.argv[0]} <device> <matrix.mtx> <n> <iterations>")
        sys.exit(1)

    device_str, backend_str, mtx_file, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) 
    kname = sys.argv[5]
    iterations= sys.argv[6]
    print(f"[INFO] Device: {device_str}, Backend: {backend_str}, Matrix: {mtx_file}, n: {n}, kname: {kname} , iterations: {iterations}")

    kernel_name_map = {
        'bstream'       :0, 
        'cstream'       :1, 
        'bstream-msplit':2, 
        'cstream-ksplit':3, 
    }

    order, etype, AName = parse_mtx_filename(mtx_file)

    ini = Inifile(f"""
[backend]
precision    = double
memory-model = large

[solver]
order = {order}

[mesh]
etype = {etype}
    """)

    if backend_str == 'cuda':
        ini.set('backend-cuda', 'gimmik-nbench', iterations)
        ini.set('backend-cuda', 'gimmik-kern', kernel_name_map.get(kname))
    elif backend_str == 'opencl':
        ini.set('backend-opencl', 'gimmik-nbench', iterations)
        ini.set('backend-opencl', 'gimmik-kern', kernel_name_map.get(kname))
    else:
        raise ValueError(f"Unsupported device: {device_str}")

    backend = get_backend(backend_str, cfg=ini)

    print(f"[INFO] Reading matrix A from {mtx_file}")
    spA = mmread(mtx_file)
    A = np.array(spA.todense(), dtype=np.float64, copy=False)
    m, k = A.shape

    B = np.ones( (k, n), dtype=np.float64)
    C = np.zeros((m, n), dtype=np.float64)

    A_be = backend.const_matrix(A, tags={'align', AName})
    B_be = backend.matrix(B.shape, B, tags={'align'})
    C_be = backend.matrix(C.shape, C, tags={'align'})

    kern = backend.kernel('mul', A_be, B_be, C_be)

    kernel_src_dir = f"kernels/pyfr/{backend_str}/p{order}/{etype}/"

    # Create directory if it doesn't exist

    os.makedirs(kernel_src_dir, exist_ok=True)
    with open(f"{kernel_src_dir}/{AName}-{kname}.cpp", "w") as f:
        f.write(backend.kernel_src)
        print(f"[INFO] Kernel source written to {f.name}")

    eff = compute_efficiency(m, k, n, avg_sec=kern.dt)

    mmtype_final = f"pyfr_{kname}"

    os.makedirs("results", exist_ok=True)
    outcsv = "results/benchmarks.csv"

    write_header = not os.path.isfile(outcsv)

    with open(outcsv, "a") as f:
        if write_header:
            f.write("device,mmtype,order,etype,OpMat,m,k,n,nnz,avg,BW\n")
        f.write(f"{device_str},{backend_str},{mmtype_final},{order},{etype},{AName},{m},{k},{n},{spA.nnz},{kern.dt:.9f},{eff:.4f}\n")

    print(f"[INFO] Results appended to {outcsv}")

if __name__ == "__main__":
    main()
