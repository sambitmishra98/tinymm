# Tiny Matrix Multiplication

This repository solely focusses on matrix multiplications of a particular form (`C = A × B`) where operator matrices `A` have 10s to 100s of rows/columns, and operand matrices `B` range from 10⁶–10⁷ rows. 
Performance of these matrices are benchmarked with vendor-provided libraries on GPUs. 
The structure of these matrices allows various optimisation techniques to be applied, such as unrolling the operator matrices and cache-blocking etc. 
Our goal is to strive for maximum possible performance of these specific matrix operations on the GPUs.

### Matrix Filename Convention
```
samples/pyfr/mats/p${i}/${etype}/${OPMAT}.mtx
```

## Benchmarking

### NVIDIA GPUs (CUDA)

```sh

export device=h100
export etype=hex
export n=1000000
export opmats="M0 M3 M6 M132 M460"
export orders=" 1 2 3 4 5 6 7 8 "
export iterations=10

nvcc -O3 -Wno-deprecated-declarations src/cuda/dense.cpp  src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/dense.exe  -lcublas
nvcc -O3 -Wno-deprecated-declarations src/cuda/sparse.cpp src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/sparse.exe -lcublas -lcusparse

for order in $orders ; do

    for opmat in $opmats ; do
        src/cuda/dense.exe  $device dense  operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
        src/cuda/sparse.exe $device sparse operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
        for kname in bstream cstream bstream-msplit cstream-ksplit ; do 
            mpirun -n 1 python3 src/kernel-generator/genpyfr.py ${device} operators/p${order}/${etype}/${opmat}.mtx ${n} $kname ${iterations} ; 
        done
    done
done

```