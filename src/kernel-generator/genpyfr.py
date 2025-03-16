#!/usr/bin/env python3

import os
import sys
import time
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

def compute_bandwidth(nnz, n, avg_sec):
    bw = 2.0 * 8.0 * nnz * n / avg_sec
    return bw

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <device> <matrix.mtx> <n>")
        sys.exit(1)

    device_str, mtx_file, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

    device_backend_map = {
        'h100' : 'cuda',
        'a100' : 'cuda',
       'mi300x':  'hip',
       'mi250' :  'hip',
       'mi100' :  'hip',
    }

    backend_str = device_backend_map.get(device_str.lower())

    order, etype, AName = parse_mtx_filename(mtx_file)

    kernel_number = 3

    ini = Inifile(f"""
[backend]
precision = double

[backend-cuda]
gimmik-kern = {kernel_number}
gimmik-nbench = 5
    """)

    backend = get_backend(backend_str, cfg=ini)

    print(f"[INFO] Reading matrix A from {mtx_file}")
    spA = mmread(mtx_file)
    A = np.array(spA.todense(), dtype=np.float64, copy=False)
    m, k = A.shape

    B = np.ones( (k, n), dtype=np.float64)
    C = np.zeros((m, n), dtype=np.float64)

    A_be = backend.const_matrix(A)
    B_be = backend.matrix(B.shape, B, tags={'align'})
    C_be = backend.matrix(C.shape, C, tags={'align'})

    kern = backend.kernel('mul', A_be, B_be, C_be)

    kernel_src = kern.src # String
    # PRint it to a file
    # Example in kernels/pyfr/p3/hex
    # Use name of matrix for help to create the file. Then add in kernel number in the end
    # to the file name
    with open(f"kernels/pyfr/p3/hex/M0-k{kernel_number}.cpp", "w") as f:
        f.write(kernel_src)


    bw = compute_bandwidth(spA.nnz, n, kern.dt)

    mmtype_final = f"{backend_str}_{kernel_number}"

    os.makedirs("results", exist_ok=True)
    outcsv = "results/benchmarks.csv"

    write_header = not os.path.isfile(outcsv)

    with open(outcsv, "a") as f:
        if write_header:
            f.write("device,mmtype,order,etype,AMatName,n,avg,BW\n")
        f.write(f"{device_str},{mmtype_final},{order},{etype},{AName},{n},{kern.dt:.9f},{bw:.6e}\n")

    print(f"[INFO] Results appended to {outcsv}")

if __name__ == "__main__":
    main()
