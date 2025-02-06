# Python OpenCV Image Processing Benchmark

Benchmarking OpenCV performance for common image processing algorithms.

## Usage

> [!WARNING]
> Make sure you have OpenCL driver installed for your GPU device.

This project uses [uv](https://docs.astral.sh/uv/) to handle dependencies.

```shell
uv sync
uv run benchmark.py [IMAGEFILE] [FOLDERPATH] [--rounds [NROUNDS] 10000]
```
