# SGEMM using Tensor Cores with error correction

## Build
This example code uses [cutf](https://github.com/enp1s0/cutf) and [mateval](https://github.com/enp1s0/mateval).
```bash
git submodule update --init
mkdir mateval/build
cd mateval/build
cmake ..
make -j
cd ../../
make
```

- In order to use `halfhalf` error correction, add `-DTC_COR` to nvcc options
- In order to use `tf32tf32` error correction, add `-DTC_COR_TF32` to nvcc options
