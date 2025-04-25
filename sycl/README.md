# Digital Image Processing Benchmark with SYCL

Benchmarking SYCL performance for image processing algorithms.

## Dependencies

- C++ linker and libraries (glibc/msvc)
- [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html)
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)
- [vcpkg](https://vcpkg.io/en/)
- GPU plugins:
  - Intel (OOBE)
  - [AMD](https://developer.codeplay.com/products/oneapi/amd/guides/)
  - [NVIDIA](https://developer.codeplay.com/products/oneapi/nvidia/guides/)

## Usage

First set environment variables based on your OS:
- For Linux on `bash`: `source /opt/intel/oneapi/setvars.sh --include-intel-llvm`
- For Windows on `cmd`: `"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --include-intel-llvm`

Some CMake description:
- You can use the preset for the `{os}` (`windows` or `linux`);
- The default configuration use Level0 and OpenCL targets;
- Use `cmake --preset {os} -DALLOW_HIP_TARGET=ON` to enable target for ROCM/HIP compatible GPUs
- Use `cmake --preset {os} -DALLOW_CUDA_TARGET=ON` to enable target for CUDA compatible GPUs

```shell
source /opt/intel/oneapi/setvars.sh --include-intel-llvm # or "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --include-intel-llvm
cmake --preset {os} # with -DALLOW_HIP_TARGET=ON or -DALLOW_CUDA_TARGET=ON
cmake --build build
./build/benchmark [IMAGEFILE] [FOLDERPATH] [[NROUNDS] 10000]
```

## Troubleshooting

- [IntelSYCLConfig.cmake fails on Windows](https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/IntelSYCLConfig-cmake-fails-on-Windows/m-p/1679888)
