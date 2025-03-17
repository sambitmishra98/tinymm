
```sh

export device=h100
export order=8
export etype=hex
export n=1000000
export iterations=5

nvcc -O3 -Wno-deprecated-declarations src/cuda/dense.cpp  src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/dense.exe  -lcublas
nvcc -O3 -Wno-deprecated-declarations src/cuda/sparse.cpp src/base.cpp src/cuda/base.cpp -x cu -o src/cuda/sparse.exe -lcublas -lcusparse

for opmat in M0 M3 M6 M132 M460 ; do
    src/cuda/dense.exe  $device dense  operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
    src/cuda/sparse.exe $device sparse operators/p${order}/${etype}/${opmat}.mtx ${n} ${iterations}
done

for opmat in M6 M132 M460 ; do
    for kname in bstream cstream bstream-msplit cstream-ksplit ; do 
        mpirun -n 1 python3 src/kernel-generator/genpyfr.py ${device} operators/p${order}/${etype}/${opmat}.mtx ${n} $kname ${iterations} ; 
    done
done

```