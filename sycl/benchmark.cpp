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
#ifdef BENCHMARK_ALLOW_CUDA_TARGET
    case sycl::backend::ext_oneapi_cuda:
        return 3;
#endif
#ifdef BENCHMARK_ALLOW_HIP_TARGET
    case sycl::backend::ext_oneapi_hip:
        return 3;
#endif
    case sycl::backend::ext_oneapi_level_zero:
        return 2;
    case sycl::backend::opencl:
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

// Kernel for image inversion
void invert_kernel(sycl::queue& q, uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels) {
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
             size_t y = idx[0];
             size_t x = idx[1];

             for (size_t c = 0; c < channels; ++c) {
                 size_t index = (y * width + x) * channels + c;
                 output[index] = 255 - input[index];
             }
         });
     }).wait_and_throw();
}

// Kernel for grayscale conversion (luminance method)
void grayscale_kernel(sycl::queue& q, uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels) {
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
             size_t y = idx[0];
             size_t x = idx[1];

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
         });
     }).wait_and_throw();
}

// Kernel for threshold operation
void threshold_kernel(sycl::queue& q, uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels, uint8_t threshold, uint8_t max_value) {
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
             size_t y = idx[0];
             size_t x = idx[1];

             for (size_t c = 0; c < channels; ++c) {
                 size_t index = (y * width + x) * channels + c;
                 output[index] = (input[index] > threshold) ? max_value : 0;
             }
         });
     }).wait_and_throw();
}

// Kernel for morphological operations (erosion/dilation)
template <MORPHOLOGY_TYPE T>
void morphology_kernel(sycl::queue& q, uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels, bool* mask, size_t mask_width, size_t mask_height) {
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
             size_t y = idx[0];
             size_t x = idx[1];

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
                         if (iy >= height) iy = 2 * height - iy - 1;
                         if (ix >= width) ix = 2 * width - ix - 1;

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
         });
     }).wait_and_throw();
}

// Kernel for convolution operations (filtering)
void convolution_kernel(sycl::queue& q, uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels, float* kernel, size_t kernel_width, size_t kernel_height) {
    q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
             size_t y = idx[0];
             size_t x = idx[1];

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
                         if (iy >= height) iy = 2 * height - iy - 1;
                         if (ix >= width) ix = 2 * width - ix - 1;

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
                 result = result < 0 ? 0 : (result > 255 ? 255 : result);

                 output[(y * width + x) * channels + c] = static_cast<uint8_t>(result);
             }
         });
     }).wait_and_throw();
}

// Gaussian blur kernel
void gaussian_blur_kernel(sycl::queue& q, uint8_t* input, uint8_t* output, size_t width, size_t height, size_t channels, float sigma, size_t kernel_size) {
    // Create Gaussian kernel
    size_t kernel_half = kernel_size / 2;
    std::vector<float> kernel_cpu(kernel_size * kernel_size);

    float sum = 0.0f;
    for (size_t y = 0; y < kernel_size; ++y) {
        for (size_t x = 0; x < kernel_size; ++x) {
            float xx = static_cast<float>(x) - kernel_half;
            float yy = static_cast<float>(y) - kernel_half;
            float value = std::exp(-(xx * xx + yy * yy) / (2.0f * sigma * sigma));
            kernel_cpu[y * kernel_size + x] = value;
            sum += value;
        }
    }

    // Normalize kernel
    for (auto& val : kernel_cpu) {
        val /= sum;
    }

    // Copy kernel to device
    float* kernel_device = sycl::malloc_device<float>(kernel_size * kernel_size, q);
    q.memcpy(kernel_device, kernel_cpu.data(), kernel_size * kernel_size * sizeof(float)).wait();

    // Call convolution kernel
    convolution_kernel(q, input, output, width, height, channels, kernel_device, kernel_size, kernel_size);

    // Free kernel
    sycl::free(kernel_device, q);
}

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
    bool cross_mask_cpu[9] = {
        // clang-format off
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
        // clang-format on
    };
    bool* d_cross_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_cross_mask, cross_mask_cpu, 9 * sizeof(bool)).wait();

    // Square mask (3x3)
    bool square_mask_cpu[9] = {
        // clang-format off
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
        // clang-format on
    };
    bool* d_square_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_square_mask, square_mask_cpu, 9 * sizeof(bool)).wait();

    // Blur 3x3 mask
    float blur_3x3_mask_cpu[9] = {
        // clang-format off
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
        // clang-format on
    };
    float* d_blur_3x3_mask = sycl::malloc_device<float>(9, q);
    q.memcpy(d_blur_3x3_mask, blur_3x3_mask_cpu, 9 * sizeof(float)).wait();

    // Blur 5x5 mask
    float blur_5x5_mask_cpu[25] = {
        // clang-format off
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f
        // clang-format on
    };
    float* d_blur_5x5_mask = sycl::malloc_device<float>(25, q);
    q.memcpy(d_blur_5x5_mask, blur_5x5_mask_cpu, 25 * sizeof(float)).wait();

    // Buffer for result retrieval
    std::vector<uint8_t> result_buffer(total_size);

    // 1. Copy operation
    {
        auto [once, times] = measure_time([&]() { q.memcpy(d_output, d_input, total_size).wait(); }, rounds);

        fmt::println("copy: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("copy-{}", filename))).string(), result);
    }

    // 2. Inversion operation
    {
        auto [once, times] = measure_time([&]() {
            invert_kernel(q, d_input, d_output, width, height, channels);
        },
                                          rounds);

        fmt::println("inversion: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("inversion-{}", filename))).string(), result);
    }

    // 3. Grayscale operation
    {
        auto [once, times] = measure_time([&]() {
            grayscale_kernel(q, d_input, d_output, width, height, channels);
        },
                                          rounds);

        fmt::println("grayscale: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("grayscale-{}", filename))).string(), result);
    }

    // 4. Threshold operation
    {
        auto [once, times] = measure_time([&]() {
            threshold_kernel(q, d_input, d_output, width, height, channels, 127, 255);
        },
                                          rounds);

        fmt::println("threshold: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("threshold-{}", filename))).string(), result);
    }

    // 5. Erode with cross mask
    {
        auto [once, times] = measure_time([&]() {
            morphology_kernel<MORPHOLOGY_TYPE::EROSION>(q, d_input, d_output, width, height, channels, d_cross_mask, 3, 3);
        },
                                          rounds);

        fmt::println("erode-cross: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("erode-cross-{}", filename))).string(), result);
    }

    // 6. Erode with square mask
    {
        auto [once, times] = measure_time([&]() {
            morphology_kernel<MORPHOLOGY_TYPE::EROSION>(q, d_input, d_output, width, height, channels, d_square_mask, 3, 3);
        },
                                          rounds);

        fmt::println("erode-square: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("erode-square-{}", filename))).string(), result);
    }

    // 7. Dilate with cross mask
    {
        auto [once, times] = measure_time([&]() {
            morphology_kernel<MORPHOLOGY_TYPE::DILATION>(q, d_input, d_output, width, height, channels, d_cross_mask, 3, 3);
        },
                                          rounds);

        fmt::println("dilate-cross: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("dilate-cross-{}", filename))).string(), result);
    }

    // 8. Dilate with square mask
    {
        auto [once, times] = measure_time([&]() {
            morphology_kernel<MORPHOLOGY_TYPE::DILATION>(q, d_input, d_output, width, height, channels, d_square_mask, 3, 3);
        },
                                          rounds);

        fmt::println("dilate-square: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("dilate-square-{}", filename))).string(), result);
    }

    // 9. Convolution with blur 3x3 mask
    {
        auto [once, times] = measure_time([&]() {
            convolution_kernel(q, d_input, d_output, width, height, channels, d_blur_3x3_mask, 3, 3);
        },
                                          rounds);

        fmt::println("convolution-blur-3x3: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("convolution-blur-3x3-{}", filename))).string(), result);
    }

    // 10. Convolution with blur 5x5 mask
    {
        auto [once, times] = measure_time([&]() {
            convolution_kernel(q, d_input, d_output, width, height, channels, d_blur_5x5_mask, 5, 5);
        },
                                          rounds);

        fmt::println("convolution-blur-5x5: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("convolution-blur-5x5-{}", filename))).string(), result);
    }

    // 11. Gaussian blur 3x3
    {
        auto [once, times] = measure_time([&]() {
            gaussian_blur_kernel(q, d_input, d_output, width, height, channels, 0.8f, 3);
        },
                                          rounds);

        fmt::println("gaussian-blur-3x3: {:.3f}s (once) | {:.3f}s ({} times)", once, times, rounds);

        // Retrieve result and save
        q.memcpy(result_buffer.data(), d_output, total_size).wait();
        cv::Mat result(height, width, image.type(), result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("gaussian-blur-3x3-{}", filename))).string(), result);
    }

    // Free device memory
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_cross_mask, q);
    sycl::free(d_square_mask, q);
    sycl::free(d_blur_3x3_mask, q);
    sycl::free(d_blur_5x5_mask, q);
}

int main(int argc, char** argv) {
    constexpr const size_t default_rounds = 1000;
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

    // Create SYCL queue with the device selector
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
    fmt::println("Compute Units: {}", q.get_device().get_info<sycl::info::device::max_compute_units>());

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
    fmt::println("Running {} benchmark rounds", rounds);
    perform_benchmark(q, image, filename, outpath, rounds);

    fmt::println("Benchmark completed successfully");
    return 0;
}
