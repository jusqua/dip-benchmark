#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// ============================================================================
// ORIGINAL IMPLEMENTATION - Using USM and raw pointers
// ============================================================================

class OriginalBlurFunctor {
private:
    uint8_t* input;
    uint8_t* output;
    size_t width, height, channels;

public:
    OriginalBlurFunctor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c)
        : input(in), output(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);
        
        if (y >= height || x >= width) return;

        // Manual border handling with mirroring
        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;
            
            // 3x3 Gaussian kernel
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;
                    
                    // Mirror boundary conditions
                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= static_cast<int>(height)) iy = 2 * height - iy - 1;
                    if (ix >= static_cast<int>(width)) ix = 2 * width - ix - 1;
                    
                    // Manual indexing into linear array
                    size_t idx = (iy * width + ix) * channels + c;
                    uint8_t pixel = input[idx];
                    
                    // Gaussian weights (simplified)
                    float weight = (ky == 0 && kx == 0) ? 0.25f : 0.125f;
                    sum += weight * pixel;
                }
            }
            
            // Manual clamping and conversion
            int result = static_cast<int>(sum + 0.5f);
            result = sycl::clamp(result, 0, 255);
            output[(y * width + x) * channels + c] = static_cast<uint8_t>(result);
        }
    }
};

void original_example(sycl::queue& q, size_t width, size_t height, size_t channels) {
    size_t total_size = width * height * channels;
    
    // USM allocation
    uint8_t* d_input = sycl::malloc_device<uint8_t>(total_size, q);
    uint8_t* d_output = sycl::malloc_device<uint8_t>(total_size, q);
    
    // Copy data (assuming host_data exists)
    // q.memcpy(d_input, host_data, total_size).wait();
    
    // Launch kernel
    sycl::range<2> global_range(height, width);
    q.parallel_for(global_range, OriginalBlurFunctor(d_input, d_output, width, height, channels)).wait();
    
    // Copy result back
    // q.memcpy(host_result, d_output, total_size).wait();
    
    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

// ============================================================================
// BINDLESS IMAGES IMPLEMENTATION - Using experimental bindless images
// ============================================================================

class BindlessBlurFunctor {
private:
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
    size_t width, height;

public:
    BindlessBlurFunctor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, size_t w, size_t h)
        : input_handle(in), output_handle(out), width(w), height(h) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);
        
        if (y >= height || x >= width) return;

        sycl::float4 sum(0.0f, 0.0f, 0.0f, 0.0f);
        
        // 3x3 Gaussian kernel - hardware texture units handle borders
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                // Direct 2D coordinate access
                sycl::int2 coord(x + kx, y + ky);
                
                // Hardware-accelerated fetch with automatic border handling
                sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
                
                // Gaussian weights (simplified)
                float weight = (ky == 0 && kx == 0) ? 0.25f : 0.125f;
                sum += weight * pixel;
            }
        }
        
        // Automatic clamping in normalized [0,1] range
        sum = sycl::clamp(sum, 0.0f, 1.0f);
        
        // Direct 2D write
        sycl::int2 coord(x, y);
        syclexp::write_image(output_handle, coord, sum);
    }
};

void bindless_example(sycl::queue& q, size_t width, size_t height) {
    // Check device capability
    if (!q.get_device().has(syclexp::aspect::ext_oneapi_bindless_images)) {
        throw std::runtime_error("Device does not support bindless images");
    }
    
    // Create image descriptors for RGBA float format
    syclexp::image_descriptor input_desc(
        {width, height}, 
        4,  // RGBA channels
        syclexp::image_type::standard, 
        syclexp::image_format::r32g32b32a32_sfloat
    );
    
    syclexp::image_descriptor output_desc(
        {width, height}, 
        4, 
        syclexp::image_type::standard, 
        syclexp::image_format::r32g32b32a32_sfloat
    );
    
    // Allocate image memory (hardware-optimized layout)
    auto input_mem = syclexp::alloc_image_mem(input_desc, q);
    auto output_mem = syclexp::alloc_image_mem(output_desc, q);
    
    // Create bindless handles
    auto input_handle = syclexp::create_image(input_mem, input_desc, q);
    auto output_handle = syclexp::create_image(output_mem, output_desc, q);
    
    // Copy data using image-aware copy
    // q.ext_oneapi_copy(host_float_data, input_mem.get_handle(), input_desc).wait();
    
    // Launch kernel
    sycl::range<2> global_range(height, width);
    q.parallel_for(global_range, BindlessBlurFunctor(input_handle, output_handle, width, height)).wait();
    
    // Copy result back
    // q.ext_oneapi_copy(output_mem.get_handle(), host_result_data, output_desc).wait();
    
    // Cleanup
    syclexp::destroy_image_handle(input_handle, q);
    syclexp::destroy_image_handle(output_handle, q);
    syclexp::free_image_mem(input_mem, q);
    syclexp::free_image_mem(output_mem, q);
}

// ============================================================================
// KEY DIFFERENCES SUMMARY
// ============================================================================

/*
1. MEMORY MANAGEMENT:
   Original: sycl::malloc_device<uint8_t>() + manual indexing
   Bindless: syclexp::alloc_image_mem() + hardware-optimized layout

2. DATA ACCESS:
   Original: input[(y * width + x) * channels + c] 
   Bindless: syclexp::fetch_image<sycl::float4>(handle, coord)

3. BORDER HANDLING:
   Original: Manual mirroring logic with bounds checking
   Bindless: Hardware texture units handle boundaries automatically

4. DATA FORMATS:
   Original: uint8_t values [0-255] with manual conversion
   Bindless: Normalized float values [0.0-1.0] 

5. MEMORY LAYOUT:
   Original: Linear array with manual stride calculation
   Bindless: GPU-optimized 2D tiled memory layout

6. PERFORMANCE:
   Original: Software-managed cache, explicit memory patterns
   Bindless: Hardware texture cache, optimized for 2D spatial locality

7. API SURFACE:
   Original: Standard SYCL 2020 USM
   Bindless: Experimental oneAPI extensions

8. DEVICE SUPPORT:
   Original: Any SYCL-compatible device with USM
   Bindless: Intel GPUs with bindless images aspect support
*/