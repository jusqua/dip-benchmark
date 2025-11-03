# Digital Image Processing Benchmark with SYCL

Benchmarking SYCL performance for image processing algorithms.

## Dependencies

### Tools

- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/) or other build system
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) or [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html)

### Libraries

- [fmt](https://fmt.dev/11.1/)
- [OpenCV](https://opencv.org/)

## Usage

### Build

> [!IMPORTANT]
> For Intel® oneAPI DPC++/C++ Compiler the usage depends on the GPU vendor see [Codeplay](https://developer.codeplay.com/) if using AMD or NVIDIA GPUs and set the correct CMAKE_CXX_FLAGS for the specific target.

```shell
cmake -G Ninja -S . -B build -D CMAKE_CXX_COMPILER=acpp # for AdaptiveCpp or -D CMAKE_CXX_COMPILER=icpx for Intel® oneAPI DPC++/C++ Compiler
cmake --build build
```

### Run

> [!IMPORTANT]
> This benchmark does set the device selector to choose the GPU device with major computing units. If you want to use a specific device see environment variables for [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/env_variables.md) or [Intel® oneAPI DPC++/C++ Compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md).

- `IMAGEFILE`: Path to the image file to be processed.
- `FOLDERPATH`: Path to the folder where the processed images will be saved.
- `NROUNDS`: Number of rounds to run the benchmark (default: 10000).

```shell
./build/benchmark [IMAGEFILE] [FOLDERPATH] [[NROUNDS] 10000]
```
