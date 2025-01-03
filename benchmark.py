import os.path
from argparse import ArgumentParser, ArgumentTypeError
from time import perf_counter
from typing import Callable

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray


def parse_image(string: str) -> tuple[MatLike, str]:
    if not os.path.isfile(string) or not cv.haveImageReader(string):
        raise ArgumentTypeError("Not a valid image file")

    return (cv.imread(string), os.path.basename(string))


def parse_dir(string: str) -> str:
    if not os.path.isdir(string):
        raise ArgumentTypeError("Not a valid directory")

    return string


def measure_time(func: Callable[[], None], rounds: int) -> tuple[float, float]:
    time_start_once = perf_counter()
    func()
    time_end_once = perf_counter()

    _round = 0
    time_start_times = perf_counter()
    while _round < rounds:
        func()
        _round += 1
    time_end_times = perf_counter()

    return (time_end_once - time_start_once, time_end_times - time_start_times)


def copy_image(input: MatLike, _output: MatLike):
    _output = input.copy()


def inversion_image(input: MatLike, output: MatLike, height: int, width: int):
    for x in range(height):
        for y in range(width):
            (r, g, b) = input[x, y]
            output[x, y] = (255 - r, 255 - g, 255 - b)


def greyscale_image(input: MatLike, output: MatLike, height: int, width: int):
    for x in range(height):
        for y in range(width):
            (r, g, b) = input[x, y].astype(np.uint16)
            mid = (r + g + b) // 3
            output[x, y] = (mid, mid, mid)


def threshold_image(input: MatLike, output: MatLike, height: int, width: int):
    for x in range(height):
        for y in range(width):
            (r, g, b) = input[x, y]
            output[x, y] = (
                255 if r > 128 else 0,
                255 if g > 128 else 0,
                255 if b > 128 else 0,
            )


def erode_image(input: MatLike, output: MatLike, height: int, width: int, mask: NDArray):
    mid = mask[0].size // 2
    (mask_height, mask_width) = mask.shape
    max_px = 255
    max_sum = 255**3

    for col in range(height):
        for row in range(width):
            r, g, b = max_px, max_px, max_px
            sum = max_sum

            for i in range(mask_height):
                for j in range(mask_width):
                    x = col + i - mid
                    y = row + j - mid

                    if x < 0 or x >= width or y < 0 or y >= height:
                        continue

                    (new_r, new_g, new_b) = input[x, y].astype(np.uint16)
                    new_sum = new_r + new_g + new_b

                    if not mask[i, j] or sum <= new_sum:
                        continue

                    r = new_r
                    g = new_g
                    b = new_b
                    sum = new_sum

            output[col, row] = (r, g, b)


def dilate_image(input: MatLike, output: MatLike, height: int, width: int, mask: NDArray):
    mid = mask[0].size // 2
    (mask_height, mask_width) = mask.shape
    min_px = 0
    min_sum = 0

    for col in range(height):
        for row in range(width):
            r, g, b = min_px, min_px, min_px
            sum = min_sum

            for i in range(mask_height):
                for j in range(mask_width):
                    x = col + i - mid
                    y = row + j - mid

                    if x < 0 or x >= width or y < 0 or y >= height:
                        continue

                    (new_r, new_g, new_b) = input[x, y].astype(np.uint16)
                    new_sum = new_r + new_g + new_b

                    if not mask[i, j] or sum >= new_sum:
                        continue

                    r = new_r
                    g = new_g
                    b = new_b
                    sum = new_sum

            output[col, row] = (r, g, b)


def convolution_image(input: MatLike, output: MatLike, height: int, width: int, mask: NDArray):
    mid = mask[0].size // 2
    (mask_height, mask_width) = mask.shape

    for col in range(height):
        for row in range(width):
            px = np.array([0, 0, 0], dtype=np.float32)

            for i in range(mask_height):
                for j in range(mask_width):
                    x = col + i - mid
                    y = row + j - mid

                    if 0 <= x < width and 0 <= y < height:
                        px += input[x, y] * mask[i, j]

            output[col, row] = np.clip(px, 0, 255).astype(np.uint8)


def gaussian_blur_3x3_image(input: MatLike, output: MatLike, height: int, width: int):
    mask = np.array([
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    ], dtype=np.float32)
    mid = mask[0].size // 2
    (mask_height, mask_width) = mask.shape

    for col in range(height):
        for row in range(width):
            px = np.array([0, 0, 0], dtype=np.float32)

            for i in range(mask_height):
                for j in range(mask_width):
                    x = col + i - mid
                    y = row + j - mid

                    if 0 <= x < width and 0 <= y < height:
                        px += input[x, y] * mask[i, j]

            output[col, row] = np.clip(px, 0, 255).astype(np.uint8)


def perform_benchmark(image: MatLike, filename: str, dir: str, rounds: int):
    height, width, _ = image.shape
    sample = image.copy()

    star_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    blur_3x3_mask = np.array([
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    ], dtype=np.float32)
    blur_5x5_mask = np.array([
        [1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0],
        [4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0],
        [6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0],
        [4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0],
        [1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0],
    ], dtype=np.float32)

    once, times = measure_time(lambda: copy_image(image, sample), rounds)
    print("copy:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"original-{filename}"), sample)

    once, times = measure_time(lambda: inversion_image(image, sample, height, width), rounds)
    print("inversion:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"inversion-{filename}"), sample)

    once, times = measure_time(lambda: greyscale_image(image, sample, height, width), rounds)
    print("grayscale:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"grayscale-{filename}"), sample)

    once, times = measure_time(lambda: threshold_image(image, sample, height, width), rounds)
    print("threshold:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"threshold-{filename}"), sample)

    once, times = measure_time(lambda: erode_image(image, sample, height, width, star_mask), rounds)
    print("erode:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"erode-{filename}"), sample)

    once, times = measure_time(lambda: dilate_image(image, sample, height, width, star_mask), rounds)
    print("dilate:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"dilate-{filename}"), sample)

    once, times = measure_time(lambda: convolution_image(image, sample, height, width, blur_3x3_mask), rounds)
    print("convolution-blur-3x3:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"convolution-blur-3x3-{filename}"), sample)

    once, times = measure_time(lambda: convolution_image(image, sample, height, width, blur_5x5_mask), rounds)
    print("convolution-blur-5x5:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"convolution-blur-5x5-{filename}"), sample)

    once, times = measure_time(lambda: gaussian_blur_3x3_image(image, sample, height, width), rounds)
    print("gaussian-blur-3x3:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"gaussian-blur-3x3-{filename}"), sample)


def main():
    parser = ArgumentParser(
        prog="benchmark.py", description="Image processing algorithms benchmark"
    )

    parser.add_argument("infile", type=parse_image, help="Path to image file")
    parser.add_argument("outdir", type=parse_dir, help="Path to image output directory")
    parser.add_argument("--rounds", type=int, default=10, help="Times to be executed, default 10")

    args = parser.parse_args()

    image: MatLike = args.infile[0]
    filename: str = args.infile[1]
    dir: str = args.outdir
    rounds: int = args.rounds

    perform_benchmark(image, filename, dir, rounds)


if __name__ == "__main__":
    main()
