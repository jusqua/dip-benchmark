# SYCL Bindless Images Benchmark

This directory contains two versions of the SYCL digital image processing benchmark:

- `benchmark.cpp` - Original implementation using raw device pointers and USM
- `benchmark_bindless.cpp` - New implementation using SYCL oneAPI experimental bindless images

## Bindless Images Implementation

The bindless images version (`benchmark_bindless.cpp`) demonstrates the use of SYCL's experimental bindless image API, which provides:

- Hardware-accelerated image operations using GPU texture units
- Optimized memory access patterns for 2D image data
- Built-in interpolation and filtering capabilities
- More efficient caching for spatial locality

### Key Differences

1. **Data Format**: Uses normalized floating-point RGBA format (0.0-1.0) instead of uint8_t
2. **Memory Management**: Uses `syclexp::alloc_image_mem()` and `syclexp::create_image()` 
3. **Data Access**: Uses `syclexp::fetch_image()` and `syclexp::write_image()` for pixel operations
4. **Border Handling**: Relies on hardware texture sampling for border cases

### API Usage

```cpp
// Create image descriptor
syclexp::image_descriptor desc({width, height}, 4, 
                               syclexp::image_type::standard, 
                               syclexp::image_format::r32g32b32a32_sfloat);

// Allocate image memory
auto image_mem = syclexp::alloc_image_mem(desc, queue);

// Create bindless handle
auto image_handle = syclexp::create_image(image_mem, desc, queue);

// Use in kernel
sycl::float4 pixel = syclexp::fetch_image<sycl::float4>(image_handle, coord);
syclexp::write_image(image_handle, coord, modified_pixel);
```

## Building

Both versions can be built using CMake:

```bash
mkdir build && cd build
cmake ..
make

# This creates two executables:
# - benchmark (original version)
# - benchmark_bindless (bindless images version)
```

## Requirements

### Original Version
- SYCL 2020 compatible compiler (Intel DPC++ or AdaptiveCpp)
- Device with USM support
- OpenCV
- fmt library

### Bindless Version
- Intel DPC++ compiler with oneAPI extensions
- Device with bindless images support (`ext_oneapi_bindless_images` aspect)
- OpenCV
- fmt library

### Supported Devices

The bindless images implementation requires:
- Intel Arc GPUs
- Intel integrated GPUs (12th gen and newer)
- Some discrete Intel GPUs with Level Zero support

Check device compatibility:
```cpp
if (device.has(syclexp::aspect::ext_oneapi_bindless_images)) {
    // Device supports bindless images
}
```

## Usage

Both executables use the same command-line interface:

```bash
./benchmark <input_image> <output_directory> [rounds]
./benchmark_bindless <input_image> <output_directory> [rounds]
```

Example:
```bash
./benchmark_bindless test.jpg results/ 1000
```

## Performance Considerations

The bindless images version may offer performance benefits for:

- **Memory Access**: Hardware texture caches optimize 2D spatial access patterns
- **Border Handling**: GPU texture units handle boundary conditions efficiently  
- **Data Layout**: RGBA format aligns with GPU memory architecture
- **Interpolation**: Hardware-accelerated sampling (future extensions)

However, it may have overhead for:
- **Data Conversion**: Converting between uint8_t and float formats
- **Memory Usage**: RGBA format uses more memory than RGB
- **API Overhead**: Experimental API may have additional validation

## Limitations

1. **Experimental API**: Subject to change in future oneAPI releases
2. **Device Support**: Limited to newer Intel GPUs
3. **Format Restrictions**: Currently uses RGBA float format only
4. **Border Handling**: Different behavior from software mirroring

## Future Enhancements

Potential improvements using bindless images:
- Sampled image access with built-in interpolation
- Multi-level image pyramids for scale-space operations
- 3D volume processing for video or multi-spectral data
- Integration with Intel's Image Processing Library (IPL)