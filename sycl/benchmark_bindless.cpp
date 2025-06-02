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
namespace syclexp = sycl::ext::oneapi::experimental;

enum MORPHOLOGY_TYPE {
    EROSION,
    DILATION
};

int computing_units_selector_v(const sycl::device& dev) {
    if (dev.has(sycl::aspect::cpu)) {
        return -1;
    }

    return dev.get_info<sycl::info::device::max_compute_units>();
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
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
    size_t width;
    size_t height;
    size_t channels;

public:
    InvertFunctor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, size_t w, size_t h, size_t c)
        : input_handle(in), output_handle(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        sycl::int2 coord(x, y);
        
        if (channels == 3) {
            sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
            sycl::float4 inverted(1.0f - pixel.r(), 1.0f - pixel.g(), 1.0f - pixel.b(), pixel.w());
            syclexp::write_image(output_handle, coord, inverted);
        } else if (channels == 4) {
            sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
            sycl::float4 inverted(1.0f - pixel.r(), 1.0f - pixel.g(), 1.0f - pixel.b(), 1.0f - pixel.w());
            syclexp::write_image(output_handle, coord, inverted);
        }
    }
};

class GrayscaleFunctor {
private:
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
    size_t width;
    size_t height;
    size_t channels;

public:
    GrayscaleFunctor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, size_t w, size_t h, size_t c)
        : input_handle(in), output_handle(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        sycl::int2 coord(x, y);
        sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
        
        // Using BT.709 coefficients for RGB->Gray conversion
        float gray = 0.2126f * pixel.r() + 0.7152f * pixel.g() + 0.0722f * pixel.b();
        sycl::float4 gray_pixel(gray, gray, gray, pixel.w());
        
        syclexp::write_image(output_handle, coord, gray_pixel);
    }
};

class ThresholdFunctor {
private:
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
    size_t width;
    size_t height;
    size_t channels;
    float threshold;
    float max_value;

public:
    ThresholdFunctor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, size_t w, size_t h, size_t c, float t, float mv)
        : input_handle(in), output_handle(out), width(w), height(h), channels(c), threshold(t), max_value(mv) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        sycl::int2 coord(x, y);
        sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
        
        sycl::float4 thresholded(
            (pixel.r() > threshold) ? max_value : 0.0f,
            (pixel.g() > threshold) ? max_value : 0.0f,
            (pixel.b() > threshold) ? max_value : 0.0f,
            pixel.w()
        );
        
        syclexp::write_image(output_handle, coord, thresholded);
    }
};

template <MORPHOLOGY_TYPE T>
class MorphologyFunctor {
private:
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
    size_t width;
    size_t height;
    size_t channels;
    bool* mask;
    size_t mask_width;
    size_t mask_height;

public:
    MorphologyFunctor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, size_t w, size_t h, size_t c, bool* m, size_t mw, size_t mh)
        : input_handle(in), output_handle(out), width(w), height(h), channels(c), mask(m), mask_width(mw), mask_height(mh) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        sycl::float4 result;
        if constexpr (T == MORPHOLOGY_TYPE::EROSION) {
            result = sycl::float4(1.0f, 1.0f, 1.0f, 1.0f);
        } else if constexpr (T == MORPHOLOGY_TYPE::DILATION) {
            result = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
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
                sycl::int2 coord(ix, iy);
                sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);

                if constexpr (T == MORPHOLOGY_TYPE::EROSION) {
                    result.r() = sycl::min(result.r(), pixel.r());
                    result.g() = sycl::min(result.g(), pixel.g());
                    result.b() = sycl::min(result.b(), pixel.b());
                } else if constexpr (T == MORPHOLOGY_TYPE::DILATION) {
                    result.r() = sycl::max(result.r(), pixel.r());
                    result.g() = sycl::max(result.g(), pixel.g());
                    result.b() = sycl::max(result.b(), pixel.b());
                }
            }
        }

        sycl::int2 coord(x, y);
        syclexp::write_image(output_handle, coord, result);
    }
};

class ConvolutionFunctor {
private:
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
    size_t width;
    size_t height;
    size_t channels;
    float* kernel;
    size_t kernel_width;
    size_t kernel_height;

public:
    ConvolutionFunctor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, size_t w, size_t h, size_t c, float* k, size_t kw, size_t kh)
        : input_handle(in), output_handle(out), width(w), height(h), channels(c), kernel(k), kernel_width(kw), kernel_height(kh) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        sycl::float4 sum(0.0f, 0.0f, 0.0f, 0.0f);

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
                sycl::int2 coord(ix, iy);
                sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);

                // Accumulate
                sum += kernel_val * pixel;
            }
        }

        // Clamp to [0, 1]
        sum.r() = sycl::clamp(sum.r(), 0.0f, 1.0f);
        sum.g() = sycl::clamp(sum.g(), 0.0f, 1.0f);
        sum.b() = sycl::clamp(sum.b(), 0.0f, 1.0f);
        sum.w() = sycl::clamp(sum.w(), 0.0f, 1.0f);

        sycl::int2 coord(x, y);
        syclexp::write_image(output_handle, coord, sum);
    }
};

class GaussianBlur3x3Functor {
private:
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
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
    GaussianBlur3x3Functor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, size_t w, size_t h, size_t c)
        : input_handle(in), output_handle(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);

        // Bounds check
        if (y >= height || x >= width) return;

        sycl::float4 sum(0.0f, 0.0f, 0.0f, 0.0f);

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
                sycl::int2 coord(ix, iy);
                sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);

                // Accumulate
                sum += kernel_val * pixel;
            }
        }

        // Clamp to [0, 1]
        sum.r() = sycl::clamp(sum.r(), 0.0f, 1.0f);
        sum.g() = sycl::clamp(sum.g(), 0.0f, 1.0f);
        sum.b() = sycl::clamp(sum.b(), 0.0f, 1.0f);
        sum.w() = sycl::clamp(sum.w(), 0.0f, 1.0f);

        sycl::int2 coord(x, y);
        syclexp::write_image(output_handle, coord, sum);
    }
};

void perform_benchmark(sycl::queue& q, const cv::Mat& image, const std::string& filename, const fs::path& outdir, size_t rounds) {
    size_t width = image.cols;
    size_t height = image.rows;
    size_t channels = image.channels();
    size_t total_size = width * height * channels;

    // Convert OpenCV image to normalized float data
    std::vector<float> float_data(width * height * 4);  // Always use RGBA
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t dst_idx = (y * width + x) * 4;
            size_t src_idx = (y * width + x) * channels;
            
            if (channels == 3) {
                float_data[dst_idx + 0] = image.data[src_idx + 2] / 255.0f; // R
                float_data[dst_idx + 1] = image.data[src_idx + 1] / 255.0f; // G
                float_data[dst_idx + 2] = image.data[src_idx + 0] / 255.0f; // B
                float_data[dst_idx + 3] = 1.0f; // A
            } else if (channels == 4) {
                float_data[dst_idx + 0] = image.data[src_idx + 2] / 255.0f; // R
                float_data[dst_idx + 1] = image.data[src_idx + 1] / 255.0f; // G
                float_data[dst_idx + 2] = image.data[src_idx + 0] / 255.0f; // B
                float_data[dst_idx + 3] = image.data[src_idx + 3] / 255.0f; // A
            }
        }
    }

    // Create image descriptors
    syclexp::image_descriptor input_desc({width, height}, 4, syclexp::image_type::standard, syclexp::image_format::r32g32b32a32_sfloat);
    syclexp::image_descriptor output_desc({width, height}, 4, syclexp::image_type::standard, syclexp::image_format::r32g32b32a32_sfloat);
    syclexp::image_descriptor aux_desc({width, height}, 4, syclexp::image_type::standard, syclexp::image_format::r32g32b32a32_sfloat);

    // Allocate device memory for images
    auto input_mem = syclexp::alloc_image_mem(input_desc, q);
    auto output_mem = syclexp::alloc_image_mem(output_desc, q);
    auto aux_mem = syclexp::alloc_image_mem(aux_desc, q);

    // Create bindless image handles
    auto input_handle = syclexp::create_image(input_mem, input_desc, q);
    auto output_handle = syclexp::create_image(output_mem, output_desc, q);
    auto aux_handle = syclexp::create_image(aux_mem, aux_desc, q);

    // Copy input data to device
    q.ext_oneapi_copy(float_data.data(), input_mem.get_handle(), input_desc).wait();

    // Prepare masks for morphological operations
    // Cross mask (3x3)
    const constexpr bool cross_mask[9] = {
        // clang-format off
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
        // clang-format on
    };
    bool* d_cross_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_cross_mask, cross_mask, 9 * sizeof(bool)).wait();

    // Square mask (3x3)
    const constexpr bool square_mask[9] = {
        // clang-format off
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
        // clang-format on
    };
    bool* d_square_mask = sycl::malloc_device<bool>(9, q);
    q.memcpy(d_square_mask, square_mask, 9 * sizeof(bool)).wait();
    
    const constexpr bool square_mask_sep[3] = { 1, 1, 1 };
    bool* d_square_mask_sep = sycl::malloc_device<bool>(3, q);
    q.memcpy(d_square_mask_sep, square_mask_sep, 3 * sizeof(bool)).wait();

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
    
    const constexpr float blur_3x3_mask_sep[] = { 1.0f / 4.0f, 1.0f / 2.0f, 1.0f / 4.0f };
    float* d_blur_3x3_mask_sep = sycl::malloc_device<float>(3, q);
    q.memcpy(d_blur_3x3_mask_sep, blur_3x3_mask_sep, 3 * sizeof(float)).wait();

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
    
    const constexpr float blur_5x5_mask_sep[] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };
    float* d_blur_5x5_mask_sep = sycl::malloc_device<float>(5, q);
    q.memcpy(d_blur_5x5_mask_sep, blur_5x5_mask_sep, 5 * sizeof(float)).wait();

    // Buffer for result retrieval
    std::vector<float> result_buffer(width * height * 4);
    std::vector<uint8_t> final_result_buffer(total_size);

    // Determine workgroup size based on device capabilities
    size_t wg_size = std::min(size_t(16), q.get_device().get_info<sycl::info::device::max_work_group_size>());
    sycl::range<2> global_range(height, width);
    sycl::range<2> local_range(wg_size, wg_size);
    while (height % local_range[0] != 0 && local_range[0] > 1) local_range[0] /= 2;
    while (width % local_range[1] != 0 && local_range[1] > 1) local_range[1] /= 2;
    sycl::nd_range<2> kernel_range(global_range, local_range);

    auto copy_result = [&]() {
        q.ext_oneapi_copy(output_mem.get_handle(), result_buffer.data(), output_desc).wait();
        
        // Convert back to uint8_t format for OpenCV
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t src_idx = (y * width + x) * 4;
                size_t dst_idx = (y * width + x) * channels;
                
                if (channels == 3) {
                    final_result_buffer[dst_idx + 0] = static_cast<uint8_t>(sycl::clamp(result_buffer[src_idx + 2] * 255.0f, 0.0f, 255.0f)); // B
                    final_result_buffer[dst_idx + 1] = static_cast<uint8_t>(sycl::clamp(result_buffer[src_idx + 1] * 255.0f, 0.0f, 255.0f)); // G
                    final_result_buffer[dst_idx + 2] = static_cast<uint8_t>(sycl::clamp(result_buffer[src_idx + 0] * 255.0f, 0.0f, 255.0f)); // R
                } else if (channels == 4) {
                    final_result_buffer[dst_idx + 0] = static_cast<uint8_t>(sycl::clamp(result_buffer[src_idx + 2] * 255.0f, 0.0f, 255.0f)); // B
                    final_result_buffer[dst_idx + 1] = static_cast<uint8_t>(sycl::clamp(result_buffer[src_idx + 1] * 255.0f, 0.0f, 255.0f)); // G
                    final_result_buffer[dst_idx + 2] = static_cast<uint8_t>(sycl::clamp(result_buffer[src_idx + 0] * 255.0f, 0.0f, 255.0f)); // R
                    final_result_buffer[dst_idx + 3] = static_cast<uint8_t>(sycl::clamp(result_buffer[src_idx + 3] * 255.0f, 0.0f, 255.0f)); // A
                }
            }
        }
    };

    std::vector<std::tuple<std::string, std::string, std::function<void()>>> operations;
    operations.push_back({ "Copy (Host to Device)", "", [&] { q.ext_oneapi_copy(float_data.data(), input_mem.get_handle(), input_desc).wait(); } });
    operations.push_back({ "Copy (Device to Host)", "", [&] { q.ext_oneapi_copy(input_mem.get_handle(), result_buffer.data(), input_desc).wait(); } });
    operations.push_back({ "Copy (Device to Device)", "copy", [&] { q.ext_oneapi_copy(input_mem.get_handle(), output_mem.get_handle(), input_desc).wait(); } });

    operations.push_back({ "Invertion", "invertion", [&] { q.parallel_for(kernel_range, InvertFunctor(input_handle, output_handle, width, height, channels)).wait(); } });
    operations.push_back({ "Grayscale", "grayscale", [&] { q.parallel_for(kernel_range, GrayscaleFunctor(input_handle, output_handle, width, height, channels)).wait(); } });
    operations.push_back({ "Threshold", "threshold", [&] { q.parallel_for(kernel_range, ThresholdFunctor(input_handle, output_handle, width, height, channels, 127.0f/255.0f, 1.0f)).wait(); } });

    operations.push_back({ "Erosion (3x3 Cross Kernel)", "erosion-cross", [&] { q.parallel_for(kernel_range, MorphologyFunctor<EROSION>(input_handle, output_handle, width, height, channels, d_cross_mask, 3, 3)).wait(); } });
    operations.push_back({ "Erosion (3x3 Square Kernel)", "erosion-square", [&] { q.parallel_for(kernel_range, MorphologyFunctor<EROSION>(input_handle, output_handle, width, height, channels, d_square_mask, 3, 3)).wait(); } });
    operations.push_back({ "Erosion (1x3+3x1 Square Kernel)", "erosion-square-separated", [&] {
        q.parallel_for(kernel_range, MorphologyFunctor<EROSION>(input_handle, aux_handle, width, height, channels, d_square_mask_sep, 1, 3)).wait();
        q.parallel_for(kernel_range, MorphologyFunctor<EROSION>(aux_handle, output_handle, width, height, channels, d_square_mask_sep, 3, 1)).wait();
    }});

    operations.push_back({ "Dilation (3x3 Cross Kernel)", "dilation-cross", [&] { q.parallel_for(kernel_range, MorphologyFunctor<DILATION>(input_handle, output_handle, width, height, channels, d_cross_mask, 3, 3)).wait(); } });
    operations.push_back({ "Dilation (3x3 Square Kernel)", "dilation-square", [&] { q.parallel_for(kernel_range, MorphologyFunctor<DILATION>(input_handle, output_handle, width, height, channels, d_square_mask, 3, 3)).wait(); } });
    operations.push_back({ "Dilation (1x3+3x1 Square Kernel)", "dilation-square-separated", [&] {
        q.parallel_for(kernel_range, MorphologyFunctor<DILATION>(input_handle, aux_handle, width, height, channels, d_square_mask_sep, 1, 3)).wait();
        q.parallel_for(kernel_range, MorphologyFunctor<DILATION>(aux_handle, output_handle, width, height, channels, d_square_mask_sep, 3, 1)).wait();
    }});

    operations.push_back({ "Convolution (3x3 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3", [&]() { q.parallel_for(kernel_range, ConvolutionFunctor(input_handle, output_handle, width, height, channels, d_blur_3x3_mask, 3, 3)).wait(); } });
    operations.push_back({ "Convolution (1x3+3x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3-separated", [&]() {
        q.parallel_for(kernel_range, ConvolutionFunctor(input_handle, aux_handle, width, height, channels, d_blur_3x3_mask_sep, 1, 3)).wait();
        q.parallel_for(kernel_range, ConvolutionFunctor(aux_handle, output_handle, width, height, channels, d_blur_3x3_mask_sep, 3, 1)).wait();
    }});

    operations.push_back({ "Convolution (5x5 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5", [&]() { q.parallel_for(kernel_range, ConvolutionFunctor(input_handle, output_handle, width, height, channels, d_blur_5x5_mask, 5, 5)).wait(); } });
    operations.push_back({ "Convolution (1x5+5x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5-separated", [&]() {
        q.parallel_for(kernel_range, ConvolutionFunctor(input_handle, aux_handle, width, height, channels, d_blur_5x5_mask_sep, 1, 5)).wait();
        q.parallel_for(kernel_range, ConvolutionFunctor(aux_handle, output_handle, width, height, channels, d_blur_5x5_mask_sep, 5, 1)).wait();
    }});

    operations.push_back({ "Gaussian Blur (3x3 Kernel)", "gaussian-blur-3x3", [&]() { q.parallel_for(kernel_range, GaussianBlur3x3Functor(input_handle, output_handle, width, height, channels)).wait(); } });
    
    auto biggest_description_length = 0;
    for (const auto& operation : operations) {
        biggest_description_length = std::max(biggest_description_length, static_cast<int>(std::get<0>(operation).length()));
    }

    for (auto& operation : operations) {
        auto [description, prefix, func] = operation;
        auto [once, times] = measure_time(func, rounds);
        fmt::println("| {: <{}} | {:10.6f}s (once) | {:10.6f}s ({} times) |", description, biggest_description_length, once, times, rounds);

        if (prefix.empty()) continue;

        copy_result();
        cv::Mat result(height, width, image.type(), final_result_buffer.data());
        cv::imwrite((outdir / fs::path(fmt::format("{}-{}", prefix, filename))).string(), result);
    }

    // Cleanup
    syclexp::destroy_image_handle(input_handle, q);
    syclexp::destroy_image_handle(output_handle, q);
    syclexp::destroy_image_handle(aux_handle, q);
    syclexp::free_image_mem(input_mem, q);
    syclexp::free_image_mem(output_mem, q);
    syclexp::free_image_mem(aux_mem, q);
    
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

    // Check for bindless images support
    if (!q.get_device().has(syclexp::aspect::ext_oneapi_bindless_images)) {
        fmt::println(stderr, "Error: Device does not support bindless images, aborting");
        return 6;
    }

    // Print device information
    fmt::println("Device: {}", q.get_device().get_info<sycl::info::device::name>());
    fmt::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

    // Load input image
    cv::Mat image = cv::imread(inpath.string());
    if (image.empty()) {
        fmt::println(stderr, "Error: Failed to load image from '{}'", inpath.string());
        return 7;
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