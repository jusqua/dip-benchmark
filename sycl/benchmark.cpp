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

enum MORPHOLOGY_TYPE {
    EROSION,
    DILATION
};

// Device selector function for prioritizing GPU backends
int priority_backend_selector_v(const sycl::device& dev) {
    if (dev.has(sycl::aspect::cpu)) {
        return 0;
    }

    switch (dev.get_backend()) {
    case sycl::backend::cuda:
    case sycl::backend::hip:
        return 3;
    case sycl::backend::ocl:
        return 2;
    case sycl::backend::level_zero:
        return 1;
    default:
        return -1;
    }
}

// Function to measure execution time
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

class InvertFunctor {
private:
    uint8_t* input;
    uint8_t* output;
    size_t width;
    size_t height;
    size_t channels;

public:
    InvertFunctor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c)
        : input(in), output(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            size_t index = (y * width + x) * channels + c;
            output[index] = 255 - input[index];
        }
    }
};

class GrayscaleFunctor {
private:
    uint8_t* input;
    uint8_t* output;
    size_t width;
    size_t height;
    size_t channels;

public:
    GrayscaleFunctor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c)
        : input(in), output(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        size_t pixel_idx = (y * width + x) * channels;
        // Using BT.709 coefficients for RGB->Gray conversion
        uint8_t gray = static_cast<uint8_t>(
            0.0722f * input[pixel_idx] +      // B
            0.7152f * input[pixel_idx + 1] +  // G
            0.2126f * input[pixel_idx + 2]    // R
        );

        // Set all channels to grayscale value
        for (size_t c = 0; c < channels; ++c) {
            output[pixel_idx + c] = gray;
        }
    }
};

class ThresholdFunctor {
private:
    uint8_t* input;
    uint8_t* output;
    size_t width;
    size_t height;
    size_t channels;
    uint8_t threshold;
    uint8_t max_value;

public:
    ThresholdFunctor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c, uint8_t t, uint8_t mv)
        : input(in), output(out), width(w), height(h), channels(c), threshold(t), max_value(mv) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            size_t index = (y * width + x) * channels + c;
            output[index] = (input[index] > threshold) ? max_value : 0;
        }
    }
};

template <MORPHOLOGY_TYPE T>
class MorphologyFunctor {
private:
    uint8_t* input;
    uint8_t* output;
    size_t width;
    size_t height;
    size_t channels;
    bool* mask;
    size_t mask_width;
    size_t mask_height;

public:
    MorphologyFunctor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c, bool* m, size_t mw, size_t mh)
        : input(in), output(out), width(w), height(h), channels(c), mask(m), mask_width(mw), mask_height(mh) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            uint8_t result;
            if constexpr (T == MORPHOLOGY_TYPE::EROSION) {
                result = 255;
            } else if constexpr (T == MORPHOLOGY_TYPE::DILATION) {
                result = 0;
            }

            for (size_t my = 0; my < mask_height; ++my) {
                for (size_t mx = 0; mx < mask_width; ++mx) {
                    // Get mask value
                    bool mask_val = mask[my * mask_width + mx];
                    if (!mask_val) continue;

                    // Compute image coordinates with mirroring at borders
                    int iy = y + my - mask_height / 2;
                    int ix = x + mx - mask_width / 2;

                    // Handle border cases with mirroring
                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= static_cast<int>(height)) iy = 2 * height - iy - 1;
                    if (ix >= static_cast<int>(width)) ix = 2 * width - ix - 1;

                    // Get pixel value
                    uint8_t pixel = input[(iy * width + ix) * channels + c];

                    if constexpr (T == MORPHOLOGY_TYPE::EROSION) {
                        result = sycl::min(result, pixel);
                    } else if constexpr (T == MORPHOLOGY_TYPE::DILATION) {
                        result = sycl::max(result, pixel);
                    }
                }
            }

            output[(y * width + x) * channels + c] = result;
        }
    }
};

class ConvolutionFunctor {
private:
    uint8_t* input;
    uint8_t* output;
    size_t width;
    size_t height;
    size_t channels;
    float* kernel;
    size_t kernel_width;
    size_t kernel_height;

public:
    ConvolutionFunctor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c, float* k, size_t kw, size_t kh)
        : input(in), output(out), width(w), height(h), channels(c), kernel(k), kernel_width(kw), kernel_height(kh) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;

            for (size_t ky = 0; ky < kernel_height; ++ky) {
                for (size_t kx = 0; kx < kernel_width; ++kx) {
                    // Compute image coordinates with mirroring at borders
                    int iy = y + ky - kernel_height / 2;
                    int ix = x + kx - kernel_width / 2;

                    // Handle border cases with mirroring
                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= static_cast<int>(height)) iy = 2 * height - iy - 1;
                    if (ix >= static_cast<int>(width)) ix = 2 * width - ix - 1;

                    // Get kernel value
                    float kernel_val = kernel[ky * kernel_width + kx];

                    // Get pixel value
                    uint8_t pixel = input[(iy * width + ix) * channels + c];

                    // Accumulate
                    sum += kernel_val * pixel;
                }
            }

            // Clamp to [0, 255]
            int result = static_cast<int>(sum + 0.5f);
            result = sycl::clamp(0, result, 255);

            output[(y * width + x) * channels + c] = static_cast<uint8_t>(result);
        }
    }
};

class GaussianBlur3x3Functor {
private:
    uint8_t* input;
    uint8_t* output;
    size_t width;
    size_t height;
    size_t channels;
    const constexpr static float kernel[9] = {
        // clang-format off
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
        // clang-format on
    };
    const constexpr static size_t kernel_width = 3;
    const constexpr static size_t kernel_height = 3;

public:
    GaussianBlur3x3Functor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c)
        : input(in), output(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;

            for (size_t ky = 0; ky < kernel_height; ++ky) {
                for (size_t kx = 0; kx < kernel_width; ++kx) {
                    // Compute image coordinates with mirroring at borders
                    int iy = y + ky - kernel_height / 2;
                    int ix = x + kx - kernel_width / 2;

                    // Handle border cases with mirroring
                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= static_cast<int>(height)) iy = 2 * height - iy - 1;
                    if (ix >= static_cast<int>(width)) ix = 2 * width - ix - 1;

                    // Get kernel value
                    float kernel_val = kernel[ky * kernel_width + kx];

                    // Get pixel value
                    uint8_t pixel = input[(iy * width + ix) * channels + c];

                    // Accumulate
                    sum += kernel_val * pixel;
                }
            }

            // Clamp to [0, 255]
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

    // Prepare device buffers
    uint8_t* d_input = sycl::malloc_device<uint8_t>(total_size, q);
    uint8_t* d_output = sycl::malloc_device<uint8_t>(total_size, q);

    // Copy input image to device
    q.memcpy(d_input, image.data, total_size).wait();

    // Prepare masks for morphological operations
    // Cross mask (3x3)
    const constexpr bool cross_mask_cpu[9] = {
        // clang-format off
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
        // clang-format on
    };
    bool* d_cross_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_cross_mask, cross_mask_cpu, 9 * sizeof(bool)).wait();

    // Square mask (3x3)
    const constexpr bool square_mask_cpu[9] = {
        // clang-format off
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
        // clang-format on
    };
    bool* d_square_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_square_mask, square_mask_cpu, 9 * sizeof(bool)).wait();

    // Blur 3x3 mask
    const constexpr float blur_3x3_mask[9] = {
        // clang-format off
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
        // clang-format on
    };
    float* d_blur_3x3_mask = sycl::malloc_device<float>(9, q);
    q.memcpy(d_blur_3x3_mask, blur_3x3_mask, 9 * sizeof(float)).wait();

    // Blur 5x5 mask
    const constexpr float blur_5x5_mask[25] = {
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

    // Buffer for result retrieval
    std::vector<uint8_t> result_buffer(total_size);

    // Determine workgroup size based on device capabilities
    size_t wg_size = std::min(size_t(16), q.get_device().get_info<sycl::info::device::max_work_group_size>());
    sycl::range<2> global_range(height, width);
    sycl::range<2> local_range(wg_size, wg_size);
    while (height % local_range[0] != 0 && local_range[0] > 1) local_range[0] /= 2;
    while (width % local_range[1] != 0 && local_range[1] > 1) local_range[1] /= 2;
    sycl::nd_range<2> kernel_range(global_range, local_range);

    std::vector<std::tuple<std::string, std::string, std::function<void()>>> operations;
    operations.push_back({ "Copy (Host to Device)", "", [&] { q.memcpy(d_input, image.data, total_size).wait(); } });
    operations.push_back({ "Copy (Device to Host)", "", [&] { q.memcpy(result_buffer.data(), d_input, total_size).wait(); } });
    operations.push_back({ "Copy (Device to Device)", "copy", [&] { q.memcpy(d_output, d_input, total_size).wait(); } });
    operations.push_back({ "Invertion", "invertion", [&] { q.parallel_for(kernel_range, InvertFunctor(d_input, d_output, width, height, channels)).wait(); } });
    operations.push_back({ "Grayscale", "grayscale", [&] { q.parallel_for(kernel_range, GrayscaleFunctor(d_input, d_output, width, height, channels)).wait(); } });
    operations.push_back({ "Threshold", "threshold", [&] { q.parallel_for(kernel_range, ThresholdFunctor(d_input, d_output, width, height, channels, 127, 255)).wait(); } });
    operations.push_back({ "Erosion (3x3 Cross Kernel)", "erosion-cross", [&] { q.parallel_for(kernel_range, MorphologyFunctor<EROSION>(d_input, d_output, width, height, channels, d_cross_mask, 3, 3)).wait(); } });
    operations.push_back({ "Erosion (3x3 Square Kernel)", "erosion-square", [&] { q.parallel_for(kernel_range, MorphologyFunctor<EROSION>(d_input, d_output, width, height, channels, d_square_mask, 3, 3)).wait(); } });
    operations.push_back({ "Dilation (3x3 Cross Kernel)", "dilation-cross", [&] { q.parallel_for(kernel_range, MorphologyFunctor<DILATION>(d_input, d_output, width, height, channels, d_cross_mask, 3, 3)).wait(); } });
    operations.push_back({ "Dilation (3x3 Square Kernel)", "dilation-square", [&] { q.parallel_for(kernel_range, MorphologyFunctor<DILATION>(d_input, d_output, width, height, channels, d_square_mask, 3, 3)).wait(); } });
    operations.push_back({ "Dilation (3x3 Square Kernel)", "dilation-square", [&] { q.parallel_for(kernel_range, MorphologyFunctor<DILATION>(d_input, d_output, width, height, channels, d_square_mask, 3, 3)).wait(); } });
    operations.push_back({ "Convolution (3x3 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3", [&]() { q.parallel_for(kernel_range, ConvolutionFunctor(d_input, d_output, width, height, channels, d_blur_3x3_mask, 3, 3)).wait(); } });
    operations.push_back({ "Convolution (5x5 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5", [&]() { q.parallel_for(kernel_range, ConvolutionFunctor(d_input, d_output, width, height, channels, d_blur_5x5_mask, 5, 5)).wait(); } });
    operations.push_back({ "3x3 Gaussian Blur", "gaussian-blur-3x3", [&]() { q.parallel_for(kernel_range, GaussianBlur3x3Functor(d_input, d_output, width, height, channels)).wait(); } });

    for (auto& operation : operations) {
        auto [description, prefix, func] = operation;
        auto [once, times] = measure_time(func, rounds);
        fmt::println("{}: {:.3f}s (once) | {:.3f}s ({} times)", description, once, times, rounds);

        if (prefix.empty()) continue;

        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("{}-{}", prefix, filename))).string(), result);
    }

    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_cross_mask, q);
    sycl::free(d_square_mask, q);
    sycl::free(d_blur_3x3_mask, q);
    sycl::free(d_blur_5x5_mask, q);
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

    auto q = sycl::queue{ priority_backend_selector_v };
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
