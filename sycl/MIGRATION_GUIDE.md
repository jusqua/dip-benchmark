# Migration Guide: From USM to SYCL Bindless Images

This guide provides step-by-step instructions for migrating from USM-based image processing to SYCL oneAPI experimental bindless images.

## Prerequisites

### Environment Setup
```bash
# Ensure you have Intel DPC++ with oneAPI extensions
source /opt/intel/oneapi/setvars.sh

# Check compiler version
icpx --version  # Should support experimental features
```

### Device Compatibility Check
```cpp
// Add this check before using bindless images
if (!queue.get_device().has(syclexp::aspect::ext_oneapi_bindless_images)) {
    throw std::runtime_error("Device does not support bindless images");
}
```

## Step 1: Header Changes

### Before (USM)
```cpp
#include <sycl/sycl.hpp>
```

### After (Bindless)
```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
```

## Step 2: Data Format Migration

### Before (USM - uint8_t)
```cpp
// Raw pointer to uint8_t data [0-255]
uint8_t* input_data;
uint8_t* output_data;
size_t total_size = width * height * channels;
```

### After (Bindless - float)
```cpp
// Normalized float data [0.0-1.0] in RGBA format
std::vector<float> float_data(width * height * 4);  // Always RGBA

// Convert uint8_t to normalized float
for (size_t i = 0; i < width * height; ++i) {
    float_data[i * 4 + 0] = uint8_data[i * channels + 2] / 255.0f; // R
    float_data[i * 4 + 1] = uint8_data[i * channels + 1] / 255.0f; // G  
    float_data[i * 4 + 2] = uint8_data[i * channels + 0] / 255.0f; // B
    float_data[i * 4 + 3] = (channels == 4) ? uint8_data[i * channels + 3] / 255.0f : 1.0f; // A
}
```

## Step 3: Memory Allocation Migration

### Before (USM)
```cpp
// Allocate USM device memory
uint8_t* d_input = sycl::malloc_device<uint8_t>(total_size, queue);
uint8_t* d_output = sycl::malloc_device<uint8_t>(total_size, queue);

// Copy data
queue.memcpy(d_input, host_data, total_size).wait();
```

### After (Bindless)
```cpp
// Create image descriptors
syclexp::image_descriptor input_desc(
    {width, height},                              // dimensions
    4,                                           // channels (RGBA)
    syclexp::image_type::standard,               // image type
    syclexp::image_format::r32g32b32a32_sfloat  // format
);

syclexp::image_descriptor output_desc({width, height}, 4, 
    syclexp::image_type::standard, 
    syclexp::image_format::r32g32b32a32_sfloat);

// Allocate image memory
auto input_mem = syclexp::alloc_image_mem(input_desc, queue);
auto output_mem = syclexp::alloc_image_mem(output_desc, queue);

// Create bindless handles
auto input_handle = syclexp::create_image(input_mem, input_desc, queue);
auto output_handle = syclexp::create_image(output_mem, output_desc, queue);

// Copy data using image-aware copy
queue.ext_oneapi_copy(float_data.data(), input_mem.get_handle(), input_desc).wait();
```

## Step 4: Kernel Function Migration

### Before (USM Kernel)
```cpp
class BlurFunctor {
    uint8_t* input;
    uint8_t* output;
    size_t width, height, channels;

public:
    BlurFunctor(uint8_t* in, uint8_t* out, size_t w, size_t h, size_t c)
        : input(in), output(out), width(w), height(h), channels(c) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);
        
        if (y >= height || x >= width) return;

        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    // Manual boundary handling
                    int iy = y + ky;
                    int ix = x + kx;
                    if (iy < 0) iy = -iy;
                    if (ix < 0) ix = -ix;
                    if (iy >= height) iy = 2 * height - iy - 1;
                    if (ix >= width) ix = 2 * width - ix - 1;
                    
                    // Manual indexing
                    size_t idx = (iy * width + ix) * channels + c;
                    sum += 0.111f * input[idx];  // Simplified kernel
                }
            }
            
            // Manual clamping and conversion
            int result = sycl::clamp(static_cast<int>(sum), 0, 255);
            output[(y * width + x) * channels + c] = static_cast<uint8_t>(result);
        }
    }
};
```

### After (Bindless Kernel)
```cpp
class BlurFunctor {
    syclexp::unsampled_image_handle input_handle;
    syclexp::unsampled_image_handle output_handle;
    size_t width, height;

public:
    BlurFunctor(syclexp::unsampled_image_handle in, syclexp::unsampled_image_handle out, 
                size_t w, size_t h)
        : input_handle(in), output_handle(out), width(w), height(h) {}

    void operator()(sycl::nd_item<2> item) const {
        size_t y = item.get_global_id(0);
        size_t x = item.get_global_id(1);
        
        if (y >= height || x >= width) return;

        sycl::float4 sum(0.0f);
        
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                // Hardware handles boundaries automatically
                sycl::int2 coord(x + kx, y + ky);
                
                // Hardware-accelerated fetch
                sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
                sum += 0.111f * pixel;  // Simplified kernel
            }
        }
        
        // Automatic clamping in [0,1] range
        sum = sycl::clamp(sum, 0.0f, 1.0f);
        
        // Direct 2D coordinate write
        sycl::int2 coord(x, y);
        syclexp::write_image(output_handle, coord, sum);
    }
};
```

## Step 5: Kernel Launch Migration

### Before (USM)
```cpp
sycl::range<2> global_range(height, width);
queue.parallel_for(global_range, 
    BlurFunctor(d_input, d_output, width, height, channels)).wait();
```

### After (Bindless)
```cpp
sycl::range<2> global_range(height, width);
queue.parallel_for(global_range, 
    BlurFunctor(input_handle, output_handle, width, height)).wait();
```

## Step 6: Result Retrieval Migration

### Before (USM)
```cpp
// Direct memcpy
std::vector<uint8_t> result(total_size);
queue.memcpy(result.data(), d_output, total_size).wait();
```

### After (Bindless)
```cpp
// Image-aware copy + format conversion
std::vector<float> float_result(width * height * 4);
queue.ext_oneapi_copy(output_mem.get_handle(), float_result.data(), output_desc).wait();

// Convert back to uint8_t if needed
std::vector<uint8_t> result(width * height * channels);
for (size_t i = 0; i < width * height; ++i) {
    result[i * channels + 0] = static_cast<uint8_t>(float_result[i * 4 + 2] * 255.0f); // B
    result[i * channels + 1] = static_cast<uint8_t>(float_result[i * 4 + 1] * 255.0f); // G
    result[i * channels + 2] = static_cast<uint8_t>(float_result[i * 4 + 0] * 255.0f); // R
    if (channels == 4) {
        result[i * channels + 3] = static_cast<uint8_t>(float_result[i * 4 + 3] * 255.0f); // A
    }
}
```

## Step 7: Cleanup Migration

### Before (USM)
```cpp
sycl::free(d_input, queue);
sycl::free(d_output, queue);
```

### After (Bindless)
```cpp
syclexp::destroy_image_handle(input_handle, queue);
syclexp::destroy_image_handle(output_handle, queue);
syclexp::free_image_mem(input_mem, queue);
syclexp::free_image_mem(output_mem, queue);
```

## Common Migration Patterns

### Threshold Operation
```cpp
// USM
output[idx] = (input[idx] > threshold) ? max_val : 0;

// Bindless
sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
sycl::float4 result(
    (pixel.r() > threshold) ? max_val : 0.0f,
    (pixel.g() > threshold) ? max_val : 0.0f, 
    (pixel.b() > threshold) ? max_val : 0.0f,
    pixel.w()
);
syclexp::write_image(output_handle, coord, result);
```

### Morphological Operations
```cpp
// USM
uint8_t result = (operation == EROSION) ? 255 : 0;
for (mask iterations) {
    size_t idx = compute_mirror_index(y+ky, x+kx);
    if (operation == EROSION) result = sycl::min(result, input[idx]);
    else result = sycl::max(result, input[idx]);
}

// Bindless  
sycl::float4 result = (operation == EROSION) ? 
    sycl::float4(1.0f) : sycl::float4(0.0f);
for (mask iterations) {
    sycl::int2 coord(x+kx, y+ky);  // Hardware handles boundaries
    sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(input_handle, coord);
    if (operation == EROSION) result = sycl::min(result, pixel);
    else result = sycl::max(result, pixel);
}
```

## Performance Optimization Tips

1. **Memory Access Patterns**: Bindless images optimize for 2D spatial locality
2. **Boundary Handling**: Let hardware handle borders instead of manual mirroring
3. **Data Format**: Consider keeping data in float format throughout pipeline
4. **Kernel Fusion**: Combine operations to minimize image memory transfers
5. **Work Group Size**: Tune for texture cache efficiency (typically 16x16 or 32x32)

## Troubleshooting

### Common Compilation Errors
- Missing bindless images header: Add `#include <sycl/ext/oneapi/experimental/bindless_images.hpp>`
- Unsupported device: Check `ext_oneapi_bindless_images` aspect
- Format mismatch: Ensure consistent image descriptor formats

### Runtime Issues
- Coordinate out of bounds: Hardware clamps automatically, check expected behavior
- Performance regression: Profile memory access patterns and cache utilization
- Image corruption: Verify RGBA channel ordering in format conversion

### Debugging Tips
- Use smaller test images for initial validation
- Compare results with reference CPU implementation
- Profile both versions to identify bottlenecks
- Check device capabilities with `sycl-ls`