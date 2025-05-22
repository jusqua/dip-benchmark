# Digital Image Processing Benchmark with VisionGL

Benchmarking VisionGL performance for image processing algorithms.

## Dependencies

### Tools

- [CMake](https://cmake.org/)
- [Clang](https://clang.llvm.org/) or other C++17 compiler
- [Ninja](https://ninja-build.org/) or other build system

### Libraries

- [fmt](https://fmt.dev/11.1/)
- [OpenCV](https://opencv.org/)
- [VisionGL](https://github.com/jusqua/visiongl) and dependencies:
  - [freeglut](https://freeglut.sourceforge.net/)
  - [GLEW](https://glew.sourceforge.net/)
  - [OpenCV](https://opencv.org/)
  - [OpenCL](https://github.com/KhronosGroup/OpenCL-SDK)

> [!NOTE]
> You can use system installed packages, build them manually or use [vcpkg](https://vcpkg.io) to provide the dependencies.

> [!NOTE]
> VisionGL must be built manually with OpenCL and OpenCV support and vcpkg does not provide it (at least at the time of writing).

## Usage

### Build

```shell
cmake -G Ninja -S . -B build -D CMAKE_CXX_COMPILER=clang++
cmake --build build
```

### Run

- `IMAGEFILE`: Path to the image file to be processed.
- `FOLDERPATH`: Path to the folder where the processed images will be saved.
- `NROUNDS`: Number of rounds to run the benchmark (default: 10000).

```shell
./build/benchmark [IMAGEFILE] [FOLDERPATH] [[NROUNDS] 10000]
```