# Digital Image Processing Benchmark with OpenCV through Python

Benchmarking OpenCV performance for image processing algorithms.

## Usage

This project uses [uv](https://docs.astral.sh/uv/) to handle dependencies.

- `IMAGEFILE`: Path to the input image file
- `FOLDERPATH`: Directory where processed images will be saved
- `--rounds`: Number of iterations for timing measurements (default: 10000)

```shell
uv sync
uv run benchmark.py [IMAGEFILE] [FOLDERPATH] [--rounds [NROUNDS]]
```
