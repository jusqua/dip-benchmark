#!/bin/bash

# Test script to compare original and bindless image benchmark versions

set -e

# Configuration
INPUT_IMAGE="${1:-test_image.jpg}"
OUTPUT_DIR="comparison_results"
ROUNDS="${2:-100}"

echo "SYCL Benchmark Comparison Test"
echo "==============================="
echo "Input image: $INPUT_IMAGE"
echo "Output directory: $OUTPUT_DIR"
echo "Rounds: $ROUNDS"
echo ""

# Check if input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Error: Input image '$INPUT_IMAGE' not found!"
    echo "Usage: $0 <input_image> [rounds]"
    echo "Example: $0 sample.jpg 1000"
    exit 1
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}/original"
mkdir -p "${OUTPUT_DIR}/bindless"

# Build both versions
echo "Building benchmarks..."
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Check if executables exist
if [ ! -f "build/benchmark" ]; then
    echo "Error: Original benchmark executable not found!"
    exit 1
fi

if [ ! -f "build/benchmark_bindless" ]; then
    echo "Error: Bindless benchmark executable not found!"
    exit 1
fi

# Run original benchmark
echo ""
echo "Running original benchmark..."
echo "============================="
time ./build/benchmark "$INPUT_IMAGE" "${OUTPUT_DIR}/original/" "$ROUNDS" > "${OUTPUT_DIR}/original_results.txt"

# Run bindless benchmark
echo ""
echo "Running bindless images benchmark..."
echo "===================================="
time ./build/benchmark_bindless "$INPUT_IMAGE" "${OUTPUT_DIR}/bindless/" "$ROUNDS" > "${OUTPUT_DIR}/bindless_results.txt"

# Compare results
echo ""
echo "Results Summary"
echo "==============="
echo ""
echo "Original benchmark results:"
cat "${OUTPUT_DIR}/original_results.txt"
echo ""
echo "Bindless benchmark results:"
cat "${OUTPUT_DIR}/bindless_results.txt"

# Check if output images were generated
echo ""
echo "Output Files Generated:"
echo "======================="
echo "Original version:"
ls -la "${OUTPUT_DIR}/original/" | grep -E '\.(jpg|png|tiff)$' | wc -l | xargs echo "  Number of images:"
echo ""
echo "Bindless version:"
ls -la "${OUTPUT_DIR}/bindless/" | grep -E '\.(jpg|png|tiff)$' | wc -l | xargs echo "  Number of images:"

# Create performance comparison
echo ""
echo "Performance Comparison"
echo "======================"
echo "Extracting timing data..."

# Extract key operations timing (simplified)
grep "Gaussian Blur" "${OUTPUT_DIR}/original_results.txt" | head -1
grep "Gaussian Blur" "${OUTPUT_DIR}/bindless_results.txt" | head -1

echo ""
echo "Comparison complete! Check ${OUTPUT_DIR}/ for detailed results."