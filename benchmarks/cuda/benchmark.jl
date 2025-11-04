using CUDA
using Images
using FileIO
using BenchmarkTools
using ArgParse
using Statistics
using StaticArrays

const BLOCK_SIZE = 16

@inline function inversion_kernel!(input, output, width, height, num_channels)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if !(1 <= x <= width && 1 <= y <= height) return end

    for c in 1:num_channels
        output[c, x, y] = 1.0f0 - input[c, x, y]
    end
    return
end

@inline function grayscale_kernel!(input, output, width, height, num_channels)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if !(1 <= x <= width && 1 <= y <= height) return end

    gray = 0.2126f0 * input[1, x, y] + 0.7152f0 * input[2, x, y] + 0.0722f0 * input[3, x, y]
    for c in 1:num_channels
        output[c, x, y] = gray
    end
    return
end

@inline function threshold_kernel!(input, output, width, height, num_channels, threshold_value=0.5f0)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if !(1 <= x <= width && 1 <= y <= height) return end

    for c in 1:num_channels
        output[c, x, y] = input[c, x, y] > threshold_value ? 1.0f0 : 0.0f0
    end
    return
end

@inline function erode_kernel!(input, output, width, height, num_channels, mask, m_half_height, m_half_width)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if !(1 <= x <= width && 1 <= y <= height) return end

    px = [0.0f0, 0.0f0, 0.0f0, 0.0f0]
    px_sum = 0.0f0
    new_px_sum = 0.0f0
    for c in 1:num_channels
        px[c] = input[c, x, y]
    end

    for my in -m_half_height:m_half_height
        for mx in -m_half_width:m_half_width
            ix = x + mx
            iy = y + my
            if !(1 <= ix <= width && 1 <= iy <= height && mask[my+m_half_height+1, mx+m_half_width+1] == 1) continue end

            for c in 1:num_channels
                new_px_sum += input[c, ix, iy]
            end
            if (new_px_sum > px_sum)
                for c in 1:num_channels
                    px[c] = input[c, ix, iy]
                end
                px_sum = new_px_sum
            end
        end
    end

    for c in 1:num_channels
        output[c, x, y] = px[c]
    end
    return
end

@inline function convolution_kernel!(input, output, width, height, num_channels, kernel, k_half_height, k_half_width)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if !(1 <= x <= width && 1 <= y <= height) return end

    sum = [0.0f0, 0.0f0, 0.0f0, 0.0f0]
    for ky in -k_half_height:k_half_height
        for kx in -k_half_width:k_half_width
            ix = x + kx
            iy = y + ky
            if !(1 <= ix <= width && 1 <= iy <= height) continue end

            for c in 1:num_channels
                sum[c] += input[c, ix, iy] * kernel[ky+k_half_height+1, kx+k_half_width+1]
            end
        end
    end

    for c in 1:num_channels
        output[c, x, y] = sum[c]
    end
    return
end

@inline function gaussian_blur_3x3_kernel!(input, output, width, height, num_channels)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if !(1 <= x <= width && 1 <= y <= height) return end

    kernel = @SMatrix [
        1.0f0/16.0f0 2.0f0/16.0f0 1.0f0/16.0f0;
        2.0f0/16.0f0 4.0f0/16.0f0 2.0f0/16.0f0;
        1.0f0/16.0f0 2.0f0/16.0f0 1.0f0/16.0f0
    ]

    sum = [0.0f0, 0.0f0, 0.0f0, 0.0f0]
    for ky in -1:1
        for kx in -1:1
            ix = x + kx
            iy = y + ky
            if !(1 <= ix <= width && 1 <= iy <= height) continue end

            for c in 1:num_channels
                sum[c] += input[c, ix, iy] * kernel[ky+2, kx+2]
            end
        end
    end

    for c in 1:num_channels
        output[c, x, y] = sum[c]
    end
    return
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
    total_time = (time_end_times - time_start_times) / 10000

    return (single_time, total_time)
end

function perform_benchmark(image, filename, outdir, rounds)
    imageview = channelview(image)
    input = Float32.(imageview)
    d_input = CuArray(input)
    d_output = similar(d_input)
    d_aux = similar(d_input)

    num_channels = size(imageview, 1)
    width = size(imageview, 2)
    height = size(imageview, 3)
    threads = (BLOCK_SIZE, BLOCK_SIZE)
    blocks = (cld(width, BLOCK_SIZE), cld(height, BLOCK_SIZE))

    d_cross_mask = CUDA.zeros(3, 3)
    d_cross_mask[2, 1:3] .= 1
    d_cross_mask[1:3, 2] .= 1

    d_square_mask = CUDA.ones(3, 3)
    d_square_mask_sep_1x3 = CUDA.ones(1, 3)
    d_square_mask_sep_3x1 = CUDA.ones(3, 1)

    d_blur_3x3 = CuArray(Float32[
        1.0/16.0 2.0/16.0 1.0/16.0;
        2.0/16.0 4.0/16.0 2.0/16.0;
        1.0/16.0 2.0/16.0 1.0/16.0
    ])

    d_blur_5x5 = CuArray(Float32[
        1.0/256.0 4.0/256.0 6.0/256.0 4.0/256.0 1.0/256.0;
        4.0/256.0 16.0/256.0 24.0/256.0 16.0/256.0 4.0/256.0;
        6.0/256.0 24.0/256.0 36.0/256.0 24.0/256.0 6.0/256.0;
        4.0/256.0 16.0/256.0 24.0/256.0 16.0/256.0 4.0/256.0;
        1.0/256.0 4.0/256.0 6.0/256.0 4.0/256.0 1.0/256.0
    ])

    d_blur_3x3_1x3 = CuArray(reshape(Float32[1.0/4.0, 1.0/2.0, 1.0/4.0], 1, 3))
    d_blur_3x3_3x1 = CuArray(reshape(Float32[1.0/4.0, 1.0/2.0, 1.0/4.0], 3, 1))
    d_blur_5x5_1x5 = CuArray(reshape(Float32[1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0], 1, 5))
    d_blur_5x5_5x1 = CuArray(reshape(Float32[1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0], 5, 1))

    operations = [
        ("Upload", nothing, () -> begin
            copyto!(d_input, input)
        end),
        ("Download", nothing, () -> begin
            copyto!(input, d_input)
        end),
        ("Copy", "copy", () -> begin
            copyto!(d_output, d_input)
        end),
        ("Inversion", "inversion", () -> begin
            @cuda blocks = blocks threads = threads inversion_kernel!(d_input, d_output, width, height, num_channels)
        end),
        ("Grayscale", "grayscale", () -> begin
            @cuda blocks = blocks threads = threads grayscale_kernel!(d_input, d_output, width, height, num_channels)
        end),
        ("Threshold", "threshold", () -> begin
            @cuda blocks = blocks threads = threads threshold_kernel!(d_input, d_output, width, height, num_channels)
        end),
        ("Erosion (3x3 Cross Kernel)", "erosion-cross", () -> begin
            @cuda blocks = blocks threads = threads erode_kernel!(d_input, d_output, width, height, num_channels, d_cross_mask, 1, 1)
        end),
        ("Erosion (3x3 Square Kernel)", "erosion-square", () -> begin
            @cuda blocks = blocks threads = threads erode_kernel!(d_input, d_output, width, height, num_channels, d_square_mask, 1, 1)
        end),
        ("Erosion (1x3+3x1 Square Kernel)", "erosion-square-separated", () -> begin
            @cuda blocks = blocks threads = threads erode_kernel!(d_input, d_aux, width, height, num_channels, d_square_mask_sep_1x3, 0, 1)
            @cuda blocks = blocks threads = threads erode_kernel!(d_aux, d_output, width, height, num_channels, d_square_mask_sep_3x1, 1, 0)
        end),
        ("Convolution (3x3 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3", () -> begin
            @cuda blocks = blocks threads = threads convolution_kernel!(d_aux, d_output, width, height, num_channels, d_blur_3x3, 1, 1)
        end),
        ("Convolution (1x3+3x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-3x3-separated", () -> begin
            @cuda blocks = blocks threads = threads convolution_kernel!(d_input, d_aux, width, height, num_channels, d_blur_3x3_1x3, 0, 1)
            @cuda blocks = blocks threads = threads convolution_kernel!(d_aux, d_output, width, height, num_channels, d_blur_3x3_3x1, 1, 0)
        end),
        ("Convolution (5x5 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5", () -> begin
            @cuda blocks = blocks threads = threads convolution_kernel!(d_aux, d_output, width, height, num_channels, d_blur_5x5, 2, 2)
        end),
        ("Convolution (1x5+5x1 Gaussian Blur Kernel)", "convolution-gaussian-blur-5x5-separated", () -> begin
            @cuda blocks = blocks threads = threads convolution_kernel!(d_input, d_aux, width, height, num_channels, d_blur_5x5_1x5, 0, 2)
            @cuda blocks = blocks threads = threads convolution_kernel!(d_aux, d_output, width, height, num_channels, d_blur_5x5_5x1, 2, 0)
        end),
        ("Gaussian Blur (3x3 Kernel)", "gaussian-blur-3x3", () -> begin
            @cuda blocks = blocks threads = threads gaussian_blur_3x3_kernel!(d_aux, d_output, width, height, num_channels)
        end)
    ]

    max_desc_length = maximum(length(desc) for (desc, _, _) in operations)

    for (description, prefix, func) in operations
        once, times = measure_time(func, rounds)

        print("| $(rpad(description, max_desc_length)) | $(lpad(string(round(once, digits=6)), 10))s (once) |")
        if rounds > 1
            print(" $(lpad(string(round(times, digits=6)), 10))s ($rounds times) |")
        end
        println()

        if (prefix === nothing) continue end

        save(joinpath(outdir, "$prefix-$filename"), colorview(RGB, Array(d_output)))
    end
end

function main()
    if !CUDA.functional()
        println("CUDA is not available. Exiting.")
        return
    end

    println("Using device: $(CUDA.name(CUDA.device()))")

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
