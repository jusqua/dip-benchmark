# Digital Image Processing Benchmark with CUDA through Julia

Benchmarking CUDA performance for image processing algorithms.

## Usage

> [!IMPORTANT]
> You need a NVIDIA GPU device with CUDA support to run this benchmark.

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. benchmark.jl [IMAGEFILE] [FOLDERPATH] [--rounds [NROUNDS] 10000]
```
