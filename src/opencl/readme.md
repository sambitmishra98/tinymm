# MM testing with OpenCL backend

All files are stored as *.cpp and compiled with gcc or icpx.
All kernels are currently created with extension *.cl.

## Directory structure
- src/cuda/kernel_launch.cpp runs any kernel stored in kernels/*

## Setup

### Environment


#### NVIDIA GPUs

```sh
qsub -I -n 1 -t 06:00:00 -q gpu_h100
setup_h100_mpich_python_pyfr

```
#### Intel GPUs

```sh
qsub -I -n 1 -t 06:00:00 -q chiatta
setup_max1550_mpich_python_pyfr 
```

### Executable specifics

```sh
export device=
export backend=opencl
export creator=gimmik
export n=1000000
export niters=10

export p=3
export etype=hex
export opmat=M0
export mmtype=bstream

```

## Compile and run commands

### src/cuda/dense.cpp

```bash
nvcc -O3 -Wno-deprecated-declarations src/cuda/dense.cpp  src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/dense.exe  -lcublas
./src/cuda/dense.exe h100 dense operators/p3/hex/M0.mtx 1000000 100
```

### src/cuda/sparse.cpp

```bash
nvcc -O3 -Wno-deprecated-declarations src/cuda/sparse.cpp src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/sparse.exe -lcublas -lcusparse
./src/cuda/sparse.exe h100 sparse operators/p3/hex/M0.mtx 1000000 100
```

### src/cuda/kernel_launch.cpp

```bash
nvcc ./src/cuda/kernel_launch.cpp ./src/cuda/base.cpp ./src/base.cpp kernels/gimmik/p3/hex/M0_cuda_bstream-msplit.cpp -x cu  -o kernels/gimmik/p3/hex/M0_cuda_bstream-msplit.exe
./kernels/gimmik/p3/hex/M0_cuda_bstream.exe h100 kernels/gimmik/p3/hex/M0_cuda_bstream.cpp 1000000 10
```

### src/cuda/kernel_launch_cublascheck.cpp

Also verify correctness of solution obtained above with cuBLAS.

```bash
nvcc ./src/cuda/kernel_launch.cpp ./src/cuda/base.cpp ./src/base.cpp kernels/gimmik/p${p}/${etype}/${opmat}_cuda_${mmtype}.cpp -x cu  -o kernels/gimmik/p${p}/${etype}/${opmat}_cuda_${mmtype}_cublascheck.exe -lcublas
source ./kernels/gimmik/p${p}/${etype}/${opmat}_cuda_${mmtype}_cublascheck.exe h100 kernels/gimmik/p${p}/${etype}/${opmat}_cuda_${mmtype}.cpp 1000000 10
```
