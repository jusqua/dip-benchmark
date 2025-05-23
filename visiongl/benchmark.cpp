#include <filesystem>
#include <functional>
#include <chrono>

#include <visiongl/cl/cl2cpp_shaders.hpp>
#include <visiongl/image.hpp>
#include <visiongl/cl/image.hpp>
#include <visiongl/context.hpp>
#include <visiongl/opencv/io.hpp>
#include <visiongl/glsl2cpp_shaders.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <opencv2/opencv.hpp>

namespace ch = std::chrono;
namespace fs = std::filesystem;

// Function to measure execution time
template <typename Func>
std::tuple<double, double> measure_time(const Func& func, size_t rounds) {
    auto time_start_once = ch::high_resolution_clock::now();
    func();
    auto time_end_once = ch::high_resolution_clock::now();
    vglClFlush();

    auto time_start_times = ch::high_resolution_clock::now();
    for (size_t i = 0; i < rounds; ++i) {
        func();
    }
    auto time_end_times = ch::high_resolution_clock::now();
    vglClFlush();

    double once_duration = ch::duration<double>(time_end_once - time_start_once).count();
    double times_duration = ch::duration<double>(time_end_times - time_start_times).count();

    return { once_duration, times_duration };
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
        fmt::println(stderr, "Error: [INPUT IMAGE] must be an image file, e.g. TIFF or DICOM");
        return 2;
    }

    fs::path outpath(argv[2]);
    if (outpath.has_filename()) {
        fmt::println(stderr, "Error: [OUTPUT PATH] must be a path to output image file");
        return 3;
    }
    
    // Create output directory if it doesn't exist
    if (!fs::exists(outpath)) {
        fs::create_directories(outpath);
    }

    auto filename = inpath.filename().string();

    vglClInit();
    vglClInteropSetFalse();
    
    auto img = vgl::opencv::load_image(inpath.string().c_str());
    vglImage3To4Channels(img);
    
    auto out = vglCreateImage(img);
    auto aux = vglCreateImage(img);

    float cross_mask[] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    float square_mask[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    float square_mask_sep[] = { 1, 1, 1 };

    float blur_3x3_mask[] = {
        // clang-format off
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
        // clang-format on
    };
    float blur_3x1_mask[] = { 1.0f / 4.0f, 1.0f / 2.0f, 1.0f / 4.0f };
    float blur_5x5_mask[] = {
        // clang-format off
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f
        // clang-format on
    };
    float blur_5x1_mask[] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };

    std::vector<std::tuple<std::string, std::string, std::function<void()>>> operations;
    operations.push_back({ "Copy (Host to Device)", "", [&] { vglClUpload(img); } });
    operations.push_back({ "Copy (Device to Host)", "", [&] { vglClDownload(img); } });
    operations.push_back({ "Copy (Device to Device)", "copy", [&] { vglClCopy(img, out); } });
    operations.push_back({ "Invertion", "invert", [&] { vglClInvert(img, out); } });
    operations.push_back({ "Grayscale", "grayscale", [&] { vglClGrayscale(img, out); } });
    operations.push_back({ "Threshold", "threshold", [&] { vglClThreshold(img, out, 0.5); } });
    operations.push_back({ "Erosion (3x3 Cross Kernel)", "erosion-cross", [&] { vglClErode(img, out, cross_mask, 3, 3); } });
    operations.push_back({ "Erosion (3x3 Square Kernel)", "erosion-square", [&] { vglClErode(img, out, square_mask, 3, 3); } });
    operations.push_back({ "Erosion (1x3+3x1 Square Separated Kernel)", "erosion-square-separated", [&] { vglClErode(img, aux, square_mask_sep, 1, 3); vglClErode(aux, out, square_mask_sep, 3, 1); } });
    operations.push_back({ "Dilation (3x3 Cross Kernel)", "dilation-cross", [&] { vglClDilate(img, out, cross_mask, 3, 3); } });
    operations.push_back({ "Dilation (3x3 Square Kernel)", "dilation-square", [&] { vglClDilate(img, out, square_mask, 3, 3); } });
    operations.push_back({ "Dilation (1x3+3x1 Square Kernel)", "dilation-square-separated", [&] { vglClDilate(img, aux, square_mask_sep, 1, 3); vglClDilate(aux, out, square_mask_sep, 3, 1); } });
    operations.push_back({ "Convolution (3x3 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3", [&] { vglClConvolution(img, out, blur_3x3_mask, 3, 3); } });
    operations.push_back({ "Convolution (1x3+3x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3-separated", [&] { vglClConvolution(img, out, blur_3x1_mask, 1, 3); vglClConvolution(img, out, blur_3x1_mask, 3, 1); } });
    operations.push_back({ "Convolution (5x5 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5", [&] { vglClConvolution(img, out, blur_5x5_mask, 5, 5); } });
    operations.push_back({ "Convolution (1x5+5x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5-separated", [&] { vglClConvolution(img, out, blur_3x1_mask, 1, 5); vglClConvolution(img, out, blur_5x1_mask, 5, 1); } });
    operations.push_back({ "Gaussian Blur (3x3 Kernel)", "gaussian-blur-3x3", [&] { vglClBlurSq3(img, out); } });

    for (auto& operation : operations) {
        auto [description, prefix, func] = operation;
        auto [once, times] = measure_time(func, rounds);
        fmt::println("{:.3f}s (once) | {:.3f}s ({} times): {}", once, times, rounds, description);

        if (prefix.empty()) continue;

        vgl::opencv::save_image((char*)(outpath / fs::path(fmt::format("{}-{}", prefix, filename))).string().c_str(), out);
    }
}
