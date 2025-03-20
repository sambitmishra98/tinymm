## Results

Script: /home/sambit.mishra/scratch/03_KERNELPERFORMANCE/tinymm-benchmarking/src/python-processing/grouped-bar.ipynb 

Data: results/benchmarks.csv


# NVIDIA GPUs

```sh

export device=max1550
export etype=hex
export n=1000000
export iterations=10

nvcc -O3 -Wno-deprecated-declarations src/cuda/dense.cpp  src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/dense.exe  -lcublas
nvcc -O3 -Wno-deprecated-declarations src/cuda/sparse.cpp src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/sparse.exe -lcublas -lcusparse

export opmats=" M0 "
export orders=" 2 "
export backends=" opencl "

for order in $orders ; do
    for opmat in $opmats ; do
        ./executables/${backends}/dense.exe  $device dense  operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
        ./executables/${backends}/sparse.exe $device sparse operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
        for kname in bstream cstream bstream-msplit cstream-ksplit ; do 
            for be in $backends ; do 
                mpirun -n 1 python3 src/kernel-generator/genpyfr.py ${device} ${be} operators/p${order}/${etype}/${opmat}.mtx ${n} $kname ${iterations} ; 
            done
        done
    done
done

```

# INTEL GPUs

```sh

export device=max1550
export etype=hex
export n=1280000
export iterations=10

export opmats=" M0 "
export orders=" 1 2 "
export backend="sycl"

for order in $orders ; do
    for opmat in $opmats ; do
        ./executables/${backend}/dense.exe  $device dense  operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
        ./executables/${backend}/sparse.exe $device sparse operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
    done
done

for order in $orders ; do
    for opmat in $opmats ; do
        for kname in "bstream" ; do 

                mkdir -p ./kernels/${backend}/p${order}/${etype}
                python3 ./src/python-processing/cuda_to_sycl.py kernels/pyfr/cuda/p${order}/${etype}/${opmat}-${kname}.cpp kernels/pyfr/${backend}/p${order}/${etype}/${opmat}_${kname}.cpp

                mkdir -p ./executables/${backend}/p${order}/${etype}
                icpx -fsycl -O3 -fsycl-device-code-split=per_source src/${backend}/kernel_launch.cpp src/${backend}/base.cpp src/base.cpp kernels/pyfr/${backend}/p${order}/${etype}/${opmat}_${kname}.cpp -o ./executables/${backend}/p${order}/${etype}/${opmat}_${kname}.exe     -Wl,--start-group       -lmkl_sycl       -lmkl_intel_lp64       -lmkl_core       -lmkl_sequential     -Wl,--end-group     -liomp5 -lpthread -lm
                ./executables/${backend}/p${order}/${etype}/${opmat}_${kname}.exe ${device} ./kernels/gimmik/p${order}/${etype}/${opmat}_${kname}.cpp ${n} ${iterations}

        done
    done
done

```