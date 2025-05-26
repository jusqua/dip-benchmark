using CUDA
using Images
using FileIO
using BenchmarkTools
using ArgParse
using Statistics

const BLOCK_SIZE = 16

function copy_kernel(input, output)
    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    function kernel!(input, output)
        c = threadIdx().x  # Channel
        tx = threadIdx().y  # Local x position
        ty = threadIdx().z  # Local y position

        # Block index
        bx = blockIdx().x
        by = blockIdx().y

        # Global position
        gx = (bx - 1) * BLOCK_SIZE + tx
        gy = (by - 1) * BLOCK_SIZE + ty

        # Process only valid pixels
        if 1 <= gx <= width && 1 <= gy <= height && c <= num_channels
            # Simple copy with coalesced memory access
            output[c, gx, gy] = input[c, gx, gy]
        end

        return nothing
    end

    threads = (num_channels, BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE))

    @cuda blocks = blocks threads = threads kernel!(input, output)
    return output
end

function inversion_kernel(input, output)
    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    # For shared memory optimization, we can load tiles of the image
    tile_width = BLOCK_SIZE
    tile_height = BLOCK_SIZE

    function kernel!(input, output)
        c = threadIdx().x  # Channel
        tx = threadIdx().y  # Local x position
        ty = threadIdx().z  # Local y position

        # Block index
        bx = blockIdx().x
        by = blockIdx().y

        # Global position
        gx = (bx - 1) * BLOCK_SIZE + tx
        gy = (by - 1) * BLOCK_SIZE + ty

        # Allocate shared memory tile
        tile = @cuDynamicSharedMem(Float32, (num_channels, tile_width, tile_height))

        # Load data into shared memory
        if 1 <= gx <= width && 1 <= gy <= height && c <= num_channels
            tile[c, tx, ty] = input[c, gx, gy]
        end

        # Ensure all threads have loaded their data
        sync_threads()

        # Process only valid pixels
        if 1 <= gx <= width && 1 <= gy <= height && c <= num_channels
            # Invert value from shared memory
            output[c, gx, gy] = 1.0f0 - tile[c, tx, ty]
        end

        return nothing
    end

    # Calculate shared memory size
    shared_mem_size = num_channels * tile_width * tile_height * sizeof(Float32)

    threads = (num_channels, BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE))

    @cuda blocks = blocks threads = threads shmem = shared_mem_size kernel!(input, output)
    return output
end

function grayscale_kernel(input, output)
    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    # For shared memory optimization, we can load tiles of the image
    tile_width = BLOCK_SIZE
    tile_height = BLOCK_SIZE

    function kernel!(input, output)
        # Thread index
        tx = threadIdx().x  # Local x position
        ty = threadIdx().y  # Local y position

        # Block index
        bx = blockIdx().x
        by = blockIdx().y

        # Global position
        gx = (bx - 1) * BLOCK_SIZE + tx
        gy = (by - 1) * BLOCK_SIZE + ty

        # Allocate shared memory for RGB values
        rgb_values = @cuDynamicSharedMem(Float32, (3, tile_width, tile_height))

        # Load RGB values into shared memory
        if 1 <= gx <= width && 1 <= gy <= height
            # Load all 3 channels for this pixel
            for c in 1:min(3, num_channels)
                rgb_values[c, tx, ty] = input[c, gx, gy]
            end
        end

        # Ensure all RGB values are loaded
        sync_threads()

        # Process only valid pixels
        if 1 <= gx <= width && 1 <= gy <= height
            # Calculate average (grayscale value)
            gray_value = 0.0f0
            for c in 1:min(3, num_channels)
                gray_value += rgb_values[c, tx, ty]
            end
            gray_value /= min(3, num_channels)

            # Set all channels to the gray value
            for c in 1:min(3, num_channels)
                output[c, gx, gy] = gray_value
            end
        end

        return nothing
    end

    # Calculate shared memory size
    shared_mem_size = 3 * tile_width * tile_height * sizeof(Float32)

    threads = (BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE))

    @cuda blocks = blocks threads = threads shmem = shared_mem_size kernel!(input, output)
    return output
end

function threshold_kernel(input, output, threshold_value=0.5f0)
    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    # For shared memory optimization, we can load tiles of the image
    tile_width = BLOCK_SIZE
    tile_height = BLOCK_SIZE

    function kernel!(input, output, threshold_value)
        c = threadIdx().x  # Channel
        tx = threadIdx().y  # Local x position
        ty = threadIdx().z  # Local y position

        # Block index
        bx = blockIdx().x
        by = blockIdx().y

        # Global position
        gx = (bx - 1) * BLOCK_SIZE + tx
        gy = (by - 1) * BLOCK_SIZE + ty

        # Allocate shared memory tile
        tile = @cuDynamicSharedMem(Float32, (num_channels, tile_width, tile_height))

        # Load data into shared memory
        if 1 <= gx <= width && 1 <= gy <= height && c <= num_channels
            tile[c, tx, ty] = input[c, gx, gy]
        end

        # Ensure all threads have loaded their data
        sync_threads()

        # Process only valid pixels
        if 1 <= gx <= width && 1 <= gy <= height && c <= num_channels
            # Apply threshold from shared memory
            output[c, gx, gy] = tile[c, tx, ty] > threshold_value ? 1.0f0 : 0.0f0
        end

        return nothing
    end

    # Calculate shared memory size
    shared_mem_size = num_channels * tile_width * tile_height * sizeof(Float32)

    threads = (num_channels, BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE))

    @cuda blocks = blocks threads = threads shmem = shared_mem_size kernel!(input, output, threshold_value)
    return output
end

function erode_kernel(input, mask, output)
    m_height, m_width = size(mask)
    m_half_height = m_height ÷ 2
    m_half_width = m_width ÷ 2

    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    function kernel!(input, mask, output)
        c = threadIdx().x
        x = blockIdx().x * blockDim().y + threadIdx().y
        y = blockIdx().y * blockDim().z + threadIdx().z

        if c <= num_channels && x <= width && y <= height
            px = input[c, x, y]
            px_sum = 0.0f0
            new_px_sum = 0.0f0

            for my in -m_half_height:m_half_height
                for mx in -m_half_width:m_half_width
                    ix = x + mx
                    iy = y + my

                    if 1 <= ix <= width && 1 <= iy <= height && mask[my+m_half_height+1, mx+m_half_width+1] == 1
                        new_px_sum = sum(input[c, ix, iy])
                        if (new_px_sum > px_sum)
                            px = input[c, ix, iy]
                            px_sum = new_px_sum
                        end
                    end
                end
            end

            output[c, x, y] = px
        end

        return nothing
    end

    threads = (num_channels, BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE), 1)

    @cuda blocks = blocks threads = threads kernel!(input, mask, output)
    return output
end

function erode_separated_kernel(input, mask1, mask2, aux, output)
    # First pass with mask1
    erode_kernel(input, mask1, aux)
    CUDA.synchronize()
    # Second pass with mask2
    erode_kernel(aux, mask2, output)
    return output
end

function dilate_kernel(input, mask, output)
    m_height, m_width = size(mask)
    m_half_height = m_height ÷ 2
    m_half_width = m_width ÷ 2

    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    function kernel!(input, mask, output)
        c = threadIdx().x
        x = blockIdx().x * blockDim().y + threadIdx().y
        y = blockIdx().y * blockDim().z + threadIdx().z

        if c <= num_channels && x <= width && y <= height
            px = input[c, x, y]
            px_sum = 0.0f0
            new_px_sum = 0.0f0

            for my in -m_half_height:m_half_height
                for mx in -m_half_width:m_half_width
                    ix = x + mx
                    iy = y + my

                    if 1 <= ix <= width && 1 <= iy <= height && mask[my+m_half_height+1, mx+m_half_width+1] == 1
                        new_px_sum = sum(input[c, ix, iy])
                        if (new_px_sum > px_sum)
                            px = input[c, ix, iy]
                            px_sum = new_px_sum
                        end
                    end
                end
            end

            output[c, x, y] = px
        end

        return nothing
    end

    threads = (num_channels, BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE), 1)

    @cuda blocks = blocks threads = threads kernel!(input, mask, output)
    return output
end

function dilate_separated_kernel(input, mask1, mask2, aux, output)
    # First pass with mask1
    dilate_kernel(input, mask1, aux)
    CUDA.synchronize()
    # Second pass with mask2
    dilate_kernel(aux, mask2, output)
    return output
end

function convolve_kernel(input, kernel, output)
    k_height, k_width = size(kernel)
    k_half_height = k_height ÷ 2
    k_half_width = k_width ÷ 2

    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    function kernel!(input, kernel, output)
        c = threadIdx().x
        x = blockIdx().x * blockDim().y + threadIdx().y
        y = blockIdx().y * blockDim().z + threadIdx().z

        if c <= num_channels && x <= width && y <= height
            sum = 0.0f0

            for ky in -k_half_height:k_half_height
                for kx in -k_half_width:k_half_width
                    ix = x + kx
                    iy = y + ky
                    if 1 <= ix <= width && 1 <= iy <= height
                        sum += input[c, ix, iy] * kernel[ky+k_half_height+1, kx+k_half_width+1]
                    end
                end
            end

            output[c, x, y] = sum
        end

        return nothing
    end

    threads = (num_channels, BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE), 1)

    @cuda blocks = blocks threads = threads kernel!(input, kernel, output)

    return output
end

function convolve_separated_kernel(input, kernel1, kernel2, aux, output)
    # First pass with kernel1
    convolve_kernel(input, kernel1, aux)
    CUDA.synchronize()
    # Second pass with kernel2
    convolve_kernel(aux, kernel2, output)
    return output
end

function gaussian_blur_3x3_kernel(input, output)
    kernel = CuArray(Float32[
        1.0/16.0 2.0/16.0 1.0/16.0;
        2.0/16.0 4.0/16.0 2.0/16.0;
        1.0/16.0 2.0/16.0 1.0/16.0
    ])

    k_height, k_width = size(kernel)
    k_half_height = k_height ÷ 2
    k_half_width = k_width ÷ 2

    num_channels = size(input, 1)
    width = size(input, 2)
    height = size(input, 3)

    function kernel!(input, output)
        c = threadIdx().x
        x = blockIdx().x * blockDim().y + threadIdx().y
        y = blockIdx().y * blockDim().z + threadIdx().z

        if c <= num_channels && x <= width && y <= height
            sum = 0.0f0

            for ky in -k_half_height:k_half_height
                for kx in -k_half_width:k_half_width
                    ix = x + kx
                    iy = y + ky
                    if 1 <= ix <= width && 1 <= iy <= height
                        sum += input[c, ix, iy] * kernel[ky+k_half_height+1, kx+k_half_width+1]
                    end
                end
            end

            output[c, x, y] = sum
        end

        return nothing
    end

    threads = (num_channels, BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE), 1)

    @cuda blocks = blocks threads = threads kernel!(input, output)
    return output
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "infile"
        help = "Path to image file"
        required = true
        "outdir"
        help = "Path to image output directory"
        required = true
        "--rounds"
        help = "Number of times to execute the benchmark"
        arg_type = Int
        default = 10000
    end

    return parse_args(s)
end

function measure_time(func, rounds)
    time_start_once = time()
    func()
    time_end_once = time()
    single_time = time_end_once - time_start_once

    time_start_times = time()
    for _ in 1:rounds
        func()
    end
    time_end_times = time()
    total_time = time_end_times - time_start_times

    return (single_time, total_time)
end

function perform_benchmark(image, filename, outdir, rounds)
    d_image = CuArray{Float32}(channelview(image))
    d_sample = similar(d_image)
    d_aux = similar(d_image)

    cross_mask = CUDA.zeros(3, 3)
    cross_mask[2, 1:3] .= 1
    cross_mask[1:3, 2] .= 1

    square_mask = CUDA.ones(3, 3)
    square_mask_sep_1x3 = CUDA.ones(1, 3)
    square_mask_sep_3x1 = CUDA.ones(3, 1)

    blur_3x3 = Float32[
        1.0/16.0 2.0/16.0 1.0/16.0;
        2.0/16.0 4.0/16.0 2.0/16.0;
        1.0/16.0 2.0/16.0 1.0/16.0
    ]
    d_blur_3x3 = CuArray(blur_3x3)

    blur_5x5 = Float32[
        1.0/256.0 4.0/256.0 6.0/256.0 4.0/256.0 1.0/256.0;
        4.0/256.0 16.0/256.0 24.0/256.0 16.0/256.0 4.0/256.0;
        6.0/256.0 24.0/256.0 36.0/256.0 24.0/256.0 6.0/256.0;
        4.0/256.0 16.0/256.0 24.0/256.0 16.0/256.0 4.0/256.0;
        1.0/256.0 4.0/256.0 6.0/256.0 4.0/256.0 1.0/256.0
    ]
    d_blur_5x5 = CuArray(blur_5x5)

    # Separated kernels
    blur_3x3_1x3 = CuArray(Float32[1.0 / 4.0 1.0 / 2.0 1.0 / 4.0])
    blur_3x3_3x1 = CuArray(Float32[1.0 / 4.0; 1.0 / 2.0; 1.0 / 4.0])
    blur_5x5_1x5 = CuArray(Float32[1.0 / 16.0 4.0 / 16.0 6.0 / 16.0 4.0 / 16.0 1.0 / 16.0])
    blur_5x5_5x1 = CuArray(Float32[1.0 / 16.0; 4.0 / 16.0; 6.0 / 16.0; 4.0 / 16.0; 1.0 / 16.0])

    # Define operations with descriptions matching Python
    operations = [
        ("Copy", "copy", () -> begin
            copy_kernel(d_image, d_sample)
            CUDA.synchronize()
        end),
        ("Inversion", "inversion", () -> begin
            inversion_kernel(d_image, d_sample)
            CUDA.synchronize()
        end),
        ("Grayscale", "grayscale", () -> begin
            grayscale_kernel(d_image, d_sample)
            CUDA.synchronize()
        end),
        ("Threshold", "threshold", () -> begin
            threshold_kernel(d_image, d_sample)
            CUDA.synchronize()
        end),
        ("Erosion (3x3 Cross Kernel)", "erosion-cross", () -> begin
            erode_kernel(d_image, cross_mask, d_sample)
            CUDA.synchronize()
        end),
        ("Erosion (3x3 Square Kernel)", "erosion-square", () -> begin
            erode_kernel(d_image, square_mask, d_sample)
            CUDA.synchronize()
        end),
        ("Erosion (1x3+3x1 Square Kernel)", "erosion-square-separated", () -> begin
            erode_separated_kernel(d_image, square_mask_sep_1x3, square_mask_sep_3x1, d_aux, d_sample)
            CUDA.synchronize()
        end),
        ("Dilation (3x3 Cross Kernel)", "dilation-cross", () -> begin
            dilate_kernel(d_image, cross_mask, d_sample)
            CUDA.synchronize()
        end),
        ("Dilation (3x3 Square Kernel)", "dilation-square", () -> begin
            dilate_kernel(d_image, square_mask, d_sample)
            CUDA.synchronize()
        end),
        ("Dilation (1x3+3x1 Square Kernel)", "dilation-square-separated", () -> begin
            dilate_separated_kernel(d_image, square_mask_sep_1x3, square_mask_sep_3x1, d_aux, d_sample)
            CUDA.synchronize()
        end),
        ("Convolution (3x3 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3", () -> begin
            convolve_kernel(d_image, d_blur_3x3, d_sample)
            CUDA.synchronize()
        end),
        ("Convolution (1x3+3x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3-separated", () -> begin
            convolve_separated_kernel(d_image, blur_3x3_1x3, blur_3x3_3x1, d_aux, d_sample)
            CUDA.synchronize()
        end),
        ("Convolution (5x5 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5", () -> begin
            convolve_kernel(d_image, d_blur_5x5, d_sample)
            CUDA.synchronize()
        end),
        ("Convolution (1x5+5x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5-separated", () -> begin
            convolve_separated_kernel(d_image, blur_5x5_1x5, blur_5x5_5x1, d_aux, d_sample)
            CUDA.synchronize()
        end),
        ("Gaussian Blur (3x3 Kernel)", "gaussian-blur-3x3", () -> begin
            gaussian_blur_3x3_kernel(d_image, d_sample)
            CUDA.synchronize()
        end)
    ]

    # Find the longest description for formatting
    max_desc_length = maximum(length(desc) for (desc, _, _) in operations)

    for (description, prefix, func) in operations
        once, times = measure_time(func, rounds)
        padded_desc = rpad(description, max_desc_length)
        println("| $padded_desc | $(lpad(string(round(once, digits=6)), 10))s (once) | $(lpad(string(round(times, digits=6)), 10))s ($rounds times) |")
        save(joinpath(outdir, "$prefix-$filename"), colorview(RGB, Array(d_sample)))
    end
end

function main()
    if CUDA.functional()
        println("CUDA is available. Using device: $(CUDA.name(CUDA.device()))")
    else
        println("CUDA is not available. Exiting.")
        return
    end

    args = parse_commandline()

    if !isfile(args["infile"])
        println("Error: Input file does not exist.")
        return
    end

    image = load(args["infile"])
    filename = basename(args["infile"])
    outdir = args["outdir"]
    rounds = args["rounds"]

    mkpath(outdir)

    perform_benchmark(image, filename, outdir, rounds)
end

main()
