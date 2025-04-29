# Digital Image Processing Benchmark with SYCL

Benchmarking SYCL performance for image processing algorithms.

## Dependencies

- C++ linker and libraries (glibc/msvc)
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md)
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)
- [vcpkg](https://vcpkg.io/en/)

## Usage

Some CMake description:
```shell
mkdir build && cd build
cmake ..
cmake --build .
./benchmark [IMAGEFILE] [FOLDERPATH] [[NROUNDS] 10000]
```
