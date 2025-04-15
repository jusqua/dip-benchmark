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

    @cuda blocks=blocks threads=threads kernel!(input, output)
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

    @cuda blocks=blocks threads=threads shmem=shared_mem_size kernel!(input, output)
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

    @cuda blocks=blocks threads=threads shmem=shared_mem_size kernel!(input, output)
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

    @cuda blocks=blocks threads=threads shmem=shared_mem_size kernel!(input, output, threshold_value)
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

    @cuda blocks=blocks threads=threads kernel!(input, mask, output)
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

    @cuda blocks=blocks threads=threads kernel!(input, mask, output)
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

    @cuda blocks=blocks threads=threads kernel!(input, kernel, output)

    return output
end

function gaussian_blur_3x3_kernel(input, output)
    kernel = CuArray(Float32[
        1.0/256.0 4.0/256.0 6.0/256.0 4.0/256.0 1.0/256.0;
        4.0/256.0 16.0/256.0 24.0/256.0 16.0/256.0 4.0/256.0;
        6.0/256.0 24.0/256.0 36.0/256.0 24.0/256.0 6.0/256.0;
        4.0/256.0 16.0/256.0 24.0/256.0 16.0/256.0 4.0/256.0;
        1.0/256.0 4.0/256.0 6.0/256.0 4.0/256.0 1.0/256.0
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

    @cuda blocks=blocks threads=threads kernel!(input, output)
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

    cross_mask = CUDA.zeros(3, 3)
    cross_mask[2, 1:3] .= 1
    cross_mask[1:3, 2] .= 1

    square_mask = CUDA.ones(3, 3)

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

    once, times = measure_time(() -> begin
            copy_kernel(d_image, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("copy: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "copy-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            inversion_kernel(d_image, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("inversion: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "inversion-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            grayscale_kernel(d_image, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("grayscale: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "grayscale-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            threshold_kernel(d_image, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("threshold: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "threshold-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            erode_kernel(d_image, cross_mask, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("erode-cross: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "erode-cross-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            erode_kernel(d_image, square_mask, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("erode-square: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "erode-square-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            dilate_kernel(d_image, cross_mask, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("dilate-cross: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "dilate-cross-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            dilate_kernel(d_image, square_mask, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("dilate-square: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "dilate-square-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            convolve_kernel(d_image, d_blur_3x3, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("convolution-blur-3x3: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "convolution-blur-3x3-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            convolve_kernel(d_image, d_blur_5x5, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("convolution-blur-5x5: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "convolution-blur-5x5-$(filename)"), colorview(RGB, Array(d_sample)))

    once, times = measure_time(() -> begin
            gaussian_blur_3x3_kernel(d_image, d_sample)
            CUDA.synchronize()
        end, rounds)
    println("gaussian-blur-3x3: $(round(once, digits=3))s (once) | $(round(times, digits=3))s ($(rounds) times)")
    save(joinpath(outdir, "gaussian-blur-3x3-$(filename)"), colorview(RGB, Array(d_sample)))
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
