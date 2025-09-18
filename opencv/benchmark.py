import os
from argparse import ArgumentParser, ArgumentTypeError
from time import perf_counter
from typing import Any, Callable

import cv2 as cv
import numpy as np


def parse_image(string: str) -> tuple[np.ndarray, str]:
    if not os.path.isfile(string) or not cv.haveImageReader(string):
        raise ArgumentTypeError("Not a valid image file")

    return (cv.imread(string), os.path.basename(string))


def parse_dir(string: str) -> str:
    if os.path.exists(string) and not os.path.isdir(string):
        raise ArgumentTypeError("Not a valid directory")

    os.makedirs(string, exist_ok=True)

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


def perform_benchmark(cpu_image: cv.typing.MatLike, filename: str, dir: str, rounds: int):
    image = cv.UMat(cpu_image)
    aux = cv.UMat(cpu_image)
    sample = cv.UMat(cpu_image)

    cross_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    square_mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    square_mask_sep_1x3 = np.array([[1, 1, 1]], dtype=np.uint8)
    square_mask_sep_3x1 = np.array([[1], [1], [1]], dtype=np.uint8)

    blur_3x3_mask = np.array(
        [
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
            [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        ],
        dtype=np.float32,
    )
    blur_3x3_1x3_mask = np.array([[1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0]], dtype=np.float32)
    blur_3x3_3x1_mask = np.array([[1.0 / 4.0], [1.0 / 2.0], [1.0 / 4.0]], dtype=np.float32)
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
    blur_5x5_1x5_mask = np.array([[1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]], dtype=np.float32)
    blur_5x5_5x1_mask = np.array([[1.0 / 16.0], [4.0 / 16.0], [6.0 / 16.0], [4.0 / 16.0], [1.0 / 16.0]], dtype=np.float32)

    operations: list[tuple[str, str, Callable[[], Any]]] = []

    def erosion_separated():
        cv.erode(image, square_mask_sep_1x3, aux)
        cv.erode(aux, square_mask_sep_3x1, sample)

    def dilation_separated():
        cv.dilate(image, square_mask_sep_1x3, aux)
        cv.dilate(aux, square_mask_sep_3x1, sample)

    def convolution_3x3_separated():
        cv.filter2D(image, -1, blur_3x3_1x3_mask, aux)
        cv.filter2D(aux, -1, blur_3x3_3x1_mask, sample)

    def convolution_5x5_separated():
        cv.filter2D(image, -1, blur_5x5_1x5_mask, aux)
        cv.filter2D(aux, -1, blur_5x5_5x1_mask, sample)

    operations.append(("Upload", "", lambda: cv.UMat(cpu_image)))
    operations.append(("Download", "", lambda: image.get()))
    operations.append(("Copy", "copy", lambda: cv.copyTo(image, sample)))
    operations.append(("Inversion", "inversion", lambda: cv.bitwise_not(image, sample)))
    operations.append(("Grayscale", "grayscale", lambda: cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR, sample)))
    operations.append(("Threshold", "threshold", lambda: cv.threshold(image, 127, 255, cv.THRESH_BINARY, sample)))
    operations.append(("Erosion (3x3 Cross Kernel)", "erosion-cross", lambda: cv.erode(image, cross_mask, sample)))
    operations.append(("Erosion (3x3 Square Kernel)", "erosion-square", lambda: cv.erode(image, square_mask, sample)))
    operations.append(("Erosion (1x3+3x1 Square Kernel)", "erosion-square-separated", erosion_separated))
    operations.append(("Dilation (3x3 Cross Kernel)", "dilation-cross", lambda: cv.dilate(image, cross_mask, sample)))
    operations.append(("Dilation (3x3 Square Kernel)", "dilation-square", lambda: cv.dilate(image, square_mask, sample)))
    operations.append(("Dilation (1x3+3x1 Square Kernel)", "dilation-square-separated", dilation_separated))
    operations.append(("Convolution (3x3 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3", lambda: cv.filter2D(image, -1, blur_3x3_mask, sample)))
    operations.append(("Convolution (1x3+3x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3-separated", convolution_3x3_separated))
    operations.append(("Convolution (5x5 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5", lambda: cv.filter2D(image, -1, blur_5x5_mask, sample)))
    operations.append(("Convolution (1x5+5x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5-separated", convolution_5x5_separated))
    operations.append(("Gaussian Blur (3x3 Kernel)", "gaussian-blur-3x3", lambda: cv.GaussianBlur(image, (3, 3), 0, sample)))

    biggest_description_length = max(len(desc) for desc, _, _ in operations)

    for description, prefix, func in operations:
        time_once, time_rounds = measure_time(func, rounds)
        print(f"| {description: <{biggest_description_length}} | {time_once:10.6f}s (once) | {time_rounds:10.6f}s ({rounds} times) |")

        output_image = sample.get()
        cv.imwrite(os.path.join(dir, f"{prefix}-{filename}"), output_image)


def main():
    if not cv.ocl.haveOpenCL():
        print("No OpenCL support found")
        return

    cv.ocl.setUseOpenCL(True)

    parser = ArgumentParser(
        prog="benchmark.py", description="Image processing algorithms benchmark with OpenCL acceleration"
    )
    parser.add_argument("infile", type=parse_image, help="Path to image file")
    parser.add_argument("outdir", type=parse_dir, help="Path to image output directory")
    parser.add_argument(
        "--rounds", type=int, default=10000, help="Times to be executed, default 10000"
    )

    args = parser.parse_args()

    image: np.ndarray = args.infile[0]
    filename: str = args.infile[1]
    dir: str = args.outdir
    rounds: int = args.rounds

    perform_benchmark(image, filename, dir, rounds)


if __name__ == "__main__":
    main()
