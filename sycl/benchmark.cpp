#include <chrono>
#include <filesystem>
#include <functional>
#include <tuple>

#include <fmt/core.h>
#include <fmt/format.h>

#include <sycl/sycl.hpp>

namespace ch = std::chrono;
namespace fs = std::filesystem;

int priority_backend_selector_v(const sycl::device &dev) {
    switch (dev.get_backend()) {
    case sycl::backend::ext_oneapi_cuda:
    case sycl::backend::ext_oneapi_hip:
        return 3;
    case sycl::backend::ext_oneapi_level_zero:
        return 2;
    case sycl::backend::opencl:
        return dev.has(sycl::aspect::gpu);
    default:
        return -1;
    }
}

int main(int argc, char **argv) {
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
        } catch (std::invalid_argument const &ex) {
            fmt::println(stderr, "Error: [ROUNDS] is an invalid argument");
            rounds = default_rounds;
        } catch (std::out_of_range const &ex) {
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

    fmt::println("Device: {}", q.get_device().get_info<sycl::info::device::name>());
    fmt::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());
    fmt::println("Compute Units: {}", q.get_device().get_info<sycl::info::device::max_compute_units>());
}
