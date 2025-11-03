#include <chrono>
#include <cstdint>
#include <filesystem>
#include <tuple>
#include <vector>
#include <string>

#include <fmt/core.h>
#include <fmt/format.h>

#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>

namespace ch = std::chrono;
namespace fs = std::filesystem;

int computing_units_selector_v(const sycl::device& dev) {
    if (dev.has(sycl::aspect::cpu)) {
        return -1;
    }

    return dev.get_info<sycl::info::device::max_compute_units>();
}

template <typename Func>
std::tuple<double, double> measure_time(const Func& func, size_t rounds) {
    auto time_start_once = ch::high_resolution_clock::now();
    func();
    auto time_end_once = ch::high_resolution_clock::now();

    auto time_start_times = ch::high_resolution_clock::now();
    for (size_t i = 0; i < rounds; ++i) {
        func();
    }
    auto time_end_times = ch::high_resolution_clock::now();

    double once_duration = ch::duration<double>(time_end_once - time_start_once).count();
    double times_duration = ch::duration<double>(time_end_times - time_start_times).count();

    return { once_duration, times_duration };
}

class Kernel {
protected:
    uint8_t* input;
    uint8_t* output;
    size_t width;
    size_t height;
    size_t channels;

public:
    Kernel(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c)
        : input(in), output(out), width(w), height(h), channels(c) {}
};

class InvertKernel : public Kernel {
public:
    using Kernel::Kernel;

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            size_t index = (y * width + x) * channels + c;
            output[index] = 255 - input[index];
        }
    }
};

class GrayscaleKernel : public Kernel {
public:
    using Kernel::Kernel;

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        if (y >= height || x >= width) return;

        size_t pixel_idx = (y * width + x) * channels;
        uint8_t gray = static_cast<uint8_t>(
            0.0722f * input[pixel_idx] +      // B
            0.7152f * input[pixel_idx + 1] +  // G
            0.2126f * input[pixel_idx + 2]    // R
        );

        for (size_t c = 0; c < channels; ++c) {
            output[pixel_idx + c] = gray;
        }
    }
};

class ThresholdKernel : public Kernel {
private:
    uint8_t threshold;
    uint8_t max_value;

public:
    ThresholdKernel(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c, uint8_t t, uint8_t mv)
        : Kernel(in, out, w, h, c), threshold(t), max_value(mv) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            size_t index = (y * width + x) * channels + c;
            output[index] = (input[index] > threshold) ? max_value : 0;
        }
    }
};

class ErosionKernel : public Kernel {
private:
    bool* mask;
    size_t mask_width;
    size_t mask_height;

public:
    ErosionKernel(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c, bool* m, size_t mw, size_t mh)
        : Kernel(in, out, w, h, c), mask(m), mask_width(mw), mask_height(mh) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            uint8_t result = 255;

            for (size_t my = 0; my < mask_height; ++my) {
                for (size_t mx = 0; mx < mask_width; ++mx) {
                    bool mask_val = mask[my * mask_width + mx];
                    if (!mask_val) continue;

                    int iy = y + my - mask_height / 2;
                    int ix = x + mx - mask_width / 2;

                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= static_cast<int>(height)) iy = 2 * height - iy - 1;
                    if (ix >= static_cast<int>(width)) ix = 2 * width - ix - 1;

                    uint8_t pixel = input[(iy * width + ix) * channels + c];

                    result = sycl::min(result, pixel);
                }
            }

            output[(y * width + x) * channels + c] = result;
        }
    }
};

class ConvolutionKernel : public Kernel {
private:
    float* kernel;
    size_t kernel_width;
    size_t kernel_height;

public:
    ConvolutionKernel(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c, float* k, size_t kw, size_t kh)
        : Kernel(in, out, w, h, c), kernel(k), kernel_width(kw), kernel_height(kh) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;

            for (size_t ky = 0; ky < kernel_height; ++ky) {
                for (size_t kx = 0; kx < kernel_width; ++kx) {
                    int iy = y + ky - kernel_height / 2;
                    int ix = x + kx - kernel_width / 2;

                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= static_cast<int>(height)) iy = 2 * height - iy - 1;
                    if (ix >= static_cast<int>(width)) ix = 2 * width - ix - 1;

                    float kernel_val = kernel[ky * kernel_width + kx];

                    uint8_t pixel = input[(iy * width + ix) * channels + c];

                    sum += kernel_val * pixel;
                }
            }

            int result = static_cast<int>(sum + 0.5f);
            result = sycl::clamp(0, result, 255);

            output[(y * width + x) * channels + c] = static_cast<uint8_t>(result);
        }
    }
};

class GaussianBlur3x3Kernel : public Kernel {
private:
    constexpr static float kernel[9] = {
        // clang-format off
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
        // clang-format on
    };
    constexpr static size_t kernel_width = 3;
    constexpr static size_t kernel_height = 3;

public:
    using Kernel::Kernel;

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;

            for (size_t ky = 0; ky < kernel_height; ++ky) {
                for (size_t kx = 0; kx < kernel_width; ++kx) {
                    int iy = y + ky - kernel_height / 2;
                    int ix = x + kx - kernel_width / 2;

                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= static_cast<int>(height)) iy = 2 * height - iy - 1;
                    if (ix >= static_cast<int>(width)) ix = 2 * width - ix - 1;

                    float kernel_val = kernel[ky * kernel_width + kx];

                    uint8_t pixel = input[(iy * width + ix) * channels + c];

                    sum += kernel_val * pixel;
                }
            }

            int result = static_cast<int>(sum + 0.5f);
            result = sycl::clamp(0, result, 255);

            output[(y * width + x) * channels + c] = static_cast<uint8_t>(result);
        }
    }
};

void perform_benchmark(sycl::queue& q, const cv::Mat& image, const std::string& filename, const fs::path& outdir, size_t rounds) {
    size_t width = image.cols;
    size_t height = image.rows;
    size_t channels = image.channels();
    size_t total_size = width * height * channels;

    uint8_t* d_input = sycl::malloc_device<uint8_t>(total_size, q);
    uint8_t* d_aux = sycl::malloc_device<uint8_t>(total_size, q);
    uint8_t* d_output = sycl::malloc_device<uint8_t>(total_size, q);

    q.memcpy(d_input, image.data, total_size).wait();

    constexpr bool cross_mask[9] = {
        // clang-format off
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
        // clang-format on
    };
    bool* d_cross_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_cross_mask, cross_mask, 9 * sizeof(bool)).wait();

    constexpr bool square_mask[9] = {
        // clang-format off
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
        // clang-format on
    };
    bool* d_square_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_square_mask, square_mask, 9 * sizeof(bool)).wait();

    constexpr bool square_mask_sep[3] = { 1, 1, 1 };
    bool* d_square_mask_sep = sycl::malloc_device<bool>(3, q);
    q.memcpy(d_square_mask_sep, square_mask_sep, 3 * sizeof(bool)).wait();

    constexpr float blur_3x3_mask[9] = {
        // clang-format off
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
        // clang-format on
    };
    float* d_blur_3x3_mask = sycl::malloc_device<float>(9, q);
    q.memcpy(d_blur_3x3_mask, blur_3x3_mask, 9 * sizeof(float)).wait();

    constexpr float blur_3x3_mask_sep[] = { 1.0f / 4.0f, 1.0f / 2.0f, 1.0f / 4.0f };
    float* d_blur_3x3_mask_sep = sycl::malloc_device<float>(3, q);
    q.memcpy(d_blur_3x3_mask_sep, blur_3x3_mask_sep, 3 * sizeof(float)).wait();

    constexpr float blur_5x5_mask[25] = {
        // clang-format off
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f
        // clang-format on
    };
    float* d_blur_5x5_mask = sycl::malloc_device<float>(25, q);
    q.memcpy(d_blur_5x5_mask, blur_5x5_mask, 25 * sizeof(float)).wait();

    constexpr float blur_5x5_mask_sep[] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };
    float* d_blur_5x5_mask_sep = sycl::malloc_device<float>(5, q);
    q.memcpy(d_blur_5x5_mask_sep, blur_5x5_mask_sep, 5 * sizeof(float)).wait();

    std::vector<uint8_t> result_buffer(total_size);

    size_t wg_size = std::min(size_t(16), q.get_device().get_info<sycl::info::device::max_work_group_size>());
    sycl::range<2> global_range(height, width);
    sycl::range<2> local_range(wg_size, wg_size);
    while (height % local_range[0] != 0 && local_range[0] > 1) local_range[0] /= 2;
    while (width % local_range[1] != 0 && local_range[1] > 1) local_range[1] /= 2;
    sycl::nd_range<2> kernel_range(global_range, local_range);

    std::vector<std::tuple<std::string, std::string, std::function<void()>>> operations;

    operations.push_back({ "Upload", "", [&] { q.memcpy(d_input, image.data, total_size).wait(); } });
    operations.push_back({ "Download", "", [&] { q.memcpy(result_buffer.data(), d_input, total_size).wait(); } });
    operations.push_back({ "Copy", "copy", [&] { q.memcpy(d_output, d_input, total_size).wait(); } });

    operations.push_back({ "Invertion", "invertion", [&] { q.parallel_for(kernel_range, InvertKernel(d_input, d_output, width, height, channels)).wait(); } });
    operations.push_back({ "Grayscale", "grayscale", [&] { q.parallel_for(kernel_range, GrayscaleKernel(d_input, d_output, width, height, channels)).wait(); } });
    operations.push_back({ "Threshold", "threshold", [&] { q.parallel_for(kernel_range, ThresholdKernel(d_input, d_output, width, height, channels, 127, 255)).wait(); } });

    operations.push_back({ "Erosion (3x3 Cross Kernel)", "erosion-cross", [&] { q.parallel_for(kernel_range, ErosionKernel(d_input, d_output, width, height, channels, d_cross_mask, 3, 3)).wait(); } });
    operations.push_back({ "Erosion (3x3 Square Kernel)", "erosion-square", [&] { q.parallel_for(kernel_range, ErosionKernel(d_input, d_output, width, height, channels, d_square_mask, 3, 3)).wait(); } });
    operations.push_back({ "Erosion (1x3+3x1 Square Kernel)", "erosion-square-separated", [&] {
        q.parallel_for(kernel_range, ErosionKernel(d_input, d_aux, width, height, channels, d_square_mask_sep, 1, 3)).wait();
        q.parallel_for(kernel_range, ErosionKernel(d_aux, d_output, width, height, channels, d_square_mask_sep, 3, 1)).wait();
    }});

    operations.push_back({ "Convolution (3x3 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3", [&]() { q.parallel_for(kernel_range, ConvolutionKernel(d_input, d_output, width, height, channels, d_blur_3x3_mask, 3, 3)).wait(); } });
    operations.push_back({ "Convolution (1x3+3x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3-separated", [&]() {
        q.parallel_for(kernel_range, ConvolutionKernel(d_input, d_aux, width, height, channels, d_blur_3x3_mask_sep, 1, 3)).wait();
        q.parallel_for(kernel_range, ConvolutionKernel(d_aux, d_output, width, height, channels, d_blur_3x3_mask_sep, 3, 1)).wait();
    }});

    operations.push_back({ "Convolution (5x5 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5", [&]() { q.parallel_for(kernel_range, ConvolutionKernel(d_input, d_output, width, height, channels, d_blur_5x5_mask, 5, 5)).wait(); } });
    operations.push_back({ "Convolution (1x5+5x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5-separated", [&]() {
        q.parallel_for(kernel_range, ConvolutionKernel(d_input, d_aux, width, height, channels, d_blur_5x5_mask_sep, 1, 5)).wait();
        q.parallel_for(kernel_range, ConvolutionKernel(d_aux, d_output, width, height, channels, d_blur_5x5_mask_sep, 5, 1)).wait();
    }});

    operations.push_back({ "Gaussian Blur (3x3 Kernel)", "gaussian-blur-3x3", [&]() { q.parallel_for(kernel_range, GaussianBlur3x3Kernel(d_input, d_output, width, height, channels)).wait(); } });

    auto biggest_description_length = 0;
    for (const auto& operation : operations) {
        biggest_description_length = std::max(biggest_description_length, static_cast<int>(std::get<0>(operation).length()));
    }

    for (auto& operation : operations) {
        auto [description, prefix, func] = operation;
        auto [once, times] = measure_time(func, rounds);
        fmt::println("| {: <{}} | {:10.6f}s (once) | {:10.6f}s ({} times) |", description, biggest_description_length, once, times, rounds);

        if (prefix.empty()) continue;

        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("{}-{}", prefix, filename))).string(), result);
    }

    sycl::free(d_input, q);
    sycl::free(d_aux, q);
    sycl::free(d_output, q);
    sycl::free(d_cross_mask, q);
    sycl::free(d_square_mask, q);
    sycl::free(d_square_mask_sep, q);
    sycl::free(d_blur_3x3_mask, q);
    sycl::free(d_blur_3x3_mask_sep, q);
    sycl::free(d_blur_5x5_mask, q);
    sycl::free(d_blur_5x5_mask_sep, q);
}

int main(int argc, char** argv) {
    constexpr const size_t default_rounds = 10000;
    size_t rounds = default_rounds;

    if (argc < 3 || argc > 4) {
        fmt::println(stderr, "Usage: {} [INPUT IMAGE] [OUTPUT PATH] [[ROUNDS] = {}]", fs::path(argv[0]).filename().string(), rounds);
        return 1;
    }

    if (argc == 4) {
        auto arg = std::string(argv[3]);
        try {
            std::size_t pos;
            rounds = std::stoi(arg, &pos);

            if (pos < arg.size()) {
                fmt::println(stderr, "Error: [ROUNDS] not a number");
                rounds = default_rounds;
            }
        } catch (std::invalid_argument const& ex) {
            fmt::println(stderr, "Error: [ROUNDS] is an invalid argument");
            rounds = default_rounds;
        } catch (std::out_of_range const& ex) {
            fmt::println(stderr, "Error: [ROUNDS] is out of range");
            rounds = default_rounds;
        }
    }

    fs::path inpath(argv[1]);
    if (!inpath.has_filename()) {
        fmt::println(stderr, "Error: [INPUT IMAGE] must be an image file, e.g. JPG, PNG, TIFF");
        return 2;
    }
    fs::path outpath(argv[2]);
    if (outpath.has_filename()) {
        fmt::println(stderr, "Error: [OUTPUT PATH] must be a path to output image file");
        return 3;
    }

    auto q = sycl::queue{ computing_units_selector_v };
    if (!q.get_device().has(sycl::aspect::gpu)) {
        fmt::println(stderr, "Error: No GPU device found, aborting");
        return 4;
    }

    auto is_usm_compatible = q.get_device().has(sycl::aspect::usm_device_allocations);
    if (!is_usm_compatible) {
        fmt::println(stderr, "Error: Device does not support USM device allocations, aborting");
        return 5;
    }

    // Print device information
    fmt::println("Device: {}", q.get_device().get_info<sycl::info::device::name>());
    fmt::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

    // Load input image
    cv::Mat image = cv::imread(inpath.string());
    if (image.empty()) {
        fmt::println(stderr, "Error: Failed to load image from '{}'", inpath.string());
        return 6;
    }

    // Create output directory if it doesn't exist
    if (!fs::exists(outpath)) {
        fs::create_directories(outpath);
    }

    // Get filename for output images
    std::string filename = inpath.filename().string();

    // Run benchmarks
    perform_benchmark(q, image, filename, outpath, rounds);

    return 0;
}
