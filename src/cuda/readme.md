# MM testing with CUDA backend

All files are stored as *.cpp and compiled with nvcc with -x cu flag.

## Directory structure
- src/cuda/dense.cpp         uses cuBLAS
- src/cuda/sparse.cpp        uses cuSPARSE (and cuBLAS for verifying results)
- src/cuda/kernel_launch.cpp runs any kernel stored in kernels/*

## Setup

### Environment

```sh
qsub -I -n 1 -t 06:00:00 -q gpu_h100
setup_h100_mpich_python_pyfr

```

### Executable specifics

```sh
export device=h100        # a100, p100, v100
export backend=cuda       # cuda, hip, opencl
export creator=pyfr       # gimmik, pyfr

export n=1000000          # Aim to saturate to peak GPU performance 
export niters=10          # Bencharking iterations

export p=1
export etype=hex
export opmat=M0           # M0, M3, M6, M132, M460

#export mmtype=bstream
#export mmtype=cstream
#export mmtype=bstream-msplit
#export mmtype=cstream
export mmtypenumber=0     # 0, 1, 2, 3 for now

```

## Direct library call

### src/cuda/dense.cpp

```bash
# Create the location of the executable
mkdir -p executables/cuda/
nvcc -O3 -Wno-deprecated-declarations -lcublas -x cu \
     src/cuda/dense.cpp  src/base.cpp src/cuda/base.cpp \
     -o executables/cuda/dense.exe
./src/cuda/dense.exe ${device} dense operators/p${p}/${etype}/${opmat}.mtx ${n} ${niters}
```

### src/cuda/sparse.cpp

```bash
mkdir -p executables/cuda/
nvcc -O3 -Wno-deprecated-declarations src/cuda/sparse.cpp src/base.cpp src/cuda/base.cpp -x cu -o executables/cuda/sparse.exe -lcublas -lcusparse
./src/cuda/sparse.exe ${device} sparse operators/p${p}/${etype}/${opmat}.mtx ${n} ${niters}
```

## Kernel creation and testing

### Kernel creation

#### Creator: GiMMiK, through PyFR

Generate only one kernel, test it and write to `results/benchmark.csv`

```bash
python3 src/kernel-generator/genpyfr.py ${device} ${backend} operators/p${p}/${etype}/${opmat}.mtx ${n} ${mmtype} ${niters}
```

#### Creator: GiMMiK 

Generate all kernels.

```bash
python3 src/kernel-generator/gengimmik.py ${mmtypenumber} ${etype} ${opmat} ${backend}
```

### Kernel benchmark

```bash
nvcc ./src/cuda/kernel_launch.cpp ./src/cuda/base.cpp ./src/base.cpp kernels/${creator}/p${p}/${etype}/${opmat}_cuda_${mmtype}.cpp -o executable/${creator}/p${p}/${etype}/${opmat}_${backend}_${mmtype}.exe -x cu -lcublas
source ./kernels/gimmik/p${p}/${etype}/${opmat}_cuda_${mmtype}.exe h100 kernels/gimmik/p${p}/${etype}/${opmat}_cuda_${mmtype}.cpp 1000000 10
```

### Kernel benchmark with cuBLAS check

Also verify correctness of solution obtained above with cuBLAS.

```bash
# Create the location of the executable
mkdir -p executable/${backend}/${creator}/p${p}/${etype}
nvcc ./src/cuda/kernel_launch_cublascheck.cpp ./src/cuda/base.cpp ./src/base.cpp kernels/${creator}/${backend}/p${p}/${etype}/${opmat}_${mmtype}.cpp -o ./executable/${creator}/${backend}/p${p}/${etype}/${opmat}_${mmtype}_cublascheck.exe -x cu -lcublas
./executable/${creator}/${backend}/p${p}/${etype}/${opmat}_${mmtype}_cublascheck.exe ${device} kernels/${creator}/${backend}/p${p}/${etype}/${opmat}_${mmtype}.cpp ${n} ${niters}
```
