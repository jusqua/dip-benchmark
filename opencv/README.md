# Digital Image Processing Benchmark with OpenCV through Python

Benchmarking OpenCV performance for image processing algorithms.

## Usage

> [!IMPORTANT]
> Make sure you have OpenCL or CUDA driver installed for your GPU device.

This project uses [uv](https://docs.astral.sh/uv/) to handle dependencies.

```shell
uv sync
uv run benchmark.py [IMAGEFILE] [FOLDERPATH] [--rounds [NROUNDS] 10000]
```
