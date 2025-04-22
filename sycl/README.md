# Digital Image Processing Benchmark with SYCL

Benchmarking SYCL performance for image processing algorithms.

## Dependencies

- C++ linker and libraries (glibc/msvc)
- [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html)
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)
- [vcpkg](https://vcpkg.io/en/)

## Usage

```shell
cmake --preset {os}-{gpu}-{build}
cmake --build build
./build/benchmark [IMAGEFILE] [FOLDERPATH] [[NROUNDS] 10000]
```

## Troubleshooting

- [IntelSYCLConfig.cmake fails on Windows](https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/IntelSYCLConfig-cmake-fails-on-Windows/m-p/1679888)
