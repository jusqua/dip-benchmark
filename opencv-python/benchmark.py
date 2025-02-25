import os.path
import sys
from argparse import ArgumentParser, ArgumentTypeError
from time import perf_counter
from typing import Any, Callable

import cv2 as cv
import numpy as np
from cv2.typing import MatLike


def parse_image(string: str) -> tuple[MatLike, str]:
    if not os.path.isfile(string) or not cv.haveImageReader(string):
        raise ArgumentTypeError("Not a valid image file")

    return (cv.imread(string), os.path.basename(string))


def parse_dir(string: str) -> str:
    if not os.path.isdir(string):
        raise ArgumentTypeError("Not a valid directory")

    return string


def measure_time(func: Callable[[], Any], rounds: int) -> tuple[float, float]:
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


def perform_benchmark(image: MatLike, filename: str, dir: str, rounds: int):
    sample = image.copy()

    star_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    blur_3x3_mask = np.array(
        [
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
            [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        ],
        dtype=np.float32,
    )
    blur_5x5_mask = np.array(
        [
            [1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0],
            [4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0],
            [6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0],
            [4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0],
            [1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0],
        ],
        dtype=np.float32,
    )

    once, times = measure_time(lambda: cv.copyTo(image, sample), rounds)
    print("copy:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"copy-{filename}"), sample)

    once, times = measure_time(lambda: cv.bitwise_not(image, sample), rounds)
    print("inversion:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"inversion-{filename}"), sample)

    once, times = measure_time(
        lambda: cv.cvtColor(
            cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR, sample
        ),
        rounds,
    )
    print("grayscale:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"grayscale-{filename}"), sample)

    once, times = measure_time(
        lambda: cv.threshold(image, 127, 255, cv.THRESH_BINARY, sample), rounds
    )
    print("threshold:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"threshold-{filename}"), sample)

    once, times = measure_time(lambda: cv.erode(image, star_mask, sample), rounds)
    print("erode:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"erode-{filename}"), sample)

    once, times = measure_time(lambda: cv.dilate(image, star_mask, sample), rounds)
    print("dilate:", f"{once:.3f}s (once)", "|", f"{times:.3f}s ({rounds} times)")
    cv.imwrite(os.path.join(dir, f"dilate-{filename}"), sample)

    once, times = measure_time(
        lambda: cv.filter2D(image, -1, blur_3x3_mask, sample), rounds
    )
    print(
        "convolution-blur-3x3:",
        f"{once:.3f}s (once)",
        "|",
        f"{times:.3f}s ({rounds} times)",
    )
    cv.imwrite(os.path.join(dir, f"convolution-blur-3x3-{filename}"), sample)

    once, times = measure_time(
        lambda: cv.filter2D(image, -1, blur_5x5_mask, sample), rounds
    )
    print(
        "convolution-blur-5x5:",
        f"{once:.3f}s (once)",
        "|",
        f"{times:.3f}s ({rounds} times)",
    )
    cv.imwrite(os.path.join(dir, f"convolution-blur-5x5-{filename}"), sample)

    once, times = measure_time(
        lambda: cv.GaussianBlur(image, (3, 3), 0, sample), rounds
    )
    print(
        "gaussian-blur-3x3:",
        f"{once:.3f}s (once)",
        "|",
        f"{times:.3f}s ({rounds} times)",
    )
    cv.imwrite(os.path.join(dir, f"gaussian-blur-3x3-{filename}"), sample)


def main():
    if cv.ocl.haveOpenCL():
        if cv.ocl.useOpenCL():
            print("OpenCL device is enabled.")
        else:
            print("OpenCL is available but not enabled. Enabling now.")
            cv.ocl.setUseOpenCL(True)
            if cv.ocl.useOpenCL():
                print("Successfully enabled OpenCL device.")
            else:
                print("Failed to enable OpenCL device.", file=sys.stderr)
                return
    else:
        print("OpenCL device is not available.", file=sys.stderr)
        return

    parser = ArgumentParser(
        prog="benchmark.py", description="Image processing algorithms benchmark"
    )

    parser.add_argument("infile", type=parse_image, help="Path to image file")
    parser.add_argument("outdir", type=parse_dir, help="Path to image output directory")
    parser.add_argument(
        "--rounds", type=int, default=10000, help="Times to be executed, default 10"
    )

    args = parser.parse_args()

    image: MatLike = args.infile[0]
    filename: str = args.infile[1]
    dir: str = args.outdir
    rounds: int = args.rounds

    perform_benchmark(image, filename, dir, rounds)


if __name__ == "__main__":
    main()
