function benchmark(infile, outdir, rounds)
% Image Processing Benchmark with GPU Acceleration
% Usage: benchmark_matlab('input_image.jpg', 'output_directory', [rounds=10000])

    if nargin < 3
        rounds = 10000;
    end

    % Check GPU availability
    try
        gpu = gpuDevice();
        fprintf('Using GPU: %s\n', gpu.Name);
    catch
        error('GPU device not available');
    end

    % Read input image
    if ~exist(infile, 'file')
        error('Input file does not exist');
    end
    [img, filename] = readImage(infile);

    % Prepare GPU data
    gpuImg = gpuArray(img);
    [~, name, ext] = fileparts(filename);
    outputBase = fullfile(outdir, [name ext]);

    % Define processing kernels
    [cross, square, blur3, blur5] = createKernels();

    % Benchmark pipeline
    operations = {
        @() copyOperation(gpuImg), 'copy';
        @() inversionOperation(gpuImg), 'inversion';
        @() grayscaleOperation(gpuImg), 'grayscale';
        @() thresholdOperation(gpuImg), 'threshold';
        @() erodeOperation(gpuImg, cross), 'erode-cross';
        @() erodeOperation(gpuImg, square), 'erode-square';
        @() dilateOperation(gpuImg, cross), 'dilate-cross';
        @() dilateOperation(gpuImg, square), 'dilate-square';
        @() convOperation(gpuImg, blur3), 'convolution-blur-3x3';
        @() convOperation(gpuImg, blur5), 'convolution-blur-5x5';
        @() gaussOperation(gpuImg), 'gaussian-blur-3x3';
    };

    for op = operations'
        [procFunc, opName] = op{:};
        [onceTime, totalTime] = timeOperation(procFunc, rounds);
        saveResult(procFunc(), outputBase, opName);
        fprintf('%s: %.3fs (once) | %.3fs (%d times)\n', ...
                pad(opName, 22), onceTime, totalTime, rounds);
    end
end

function [cross, square, blur3, blur5] = createKernels()
    % Create processing kernels with double precision
    cross = strel('arbitrary', [0 1 0; 1 1 1; 0 1 0]);
    square = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]);

    % Double-precision kernel definitions
    blur3 = gpuArray(double([1  2  1;
                            2  4  2;
                            1  2  1]/16));

    blur5 = gpuArray(double([1  4  6  4 1;
                            4 16 24 16 4;
                            6 24 36 24 6;
                            4 16 24 16 4;
                            1  4  6  4 1]/256));
end

function [img, filename] = readImage(path)
    % Read and validate image
    img = imread(path);

    % Extract filename using MATLAB's fileparts
    [~, name, ext] = fileparts(path);
    filename = [name ext]; % Combine name and extension
end

function [onceTime, totalTime] = timeOperation(operation, rounds)
    % Time operation execution
    tic; % Warm-up
    operation();
    wait(gpuDevice);

    tic;
    result = operation();
    wait(gpuDevice);
    onceTime = toc;

    tic;
    for i = 1:rounds
        operation();
    end
    wait(gpuDevice);
    totalTime = toc;
end

function saveResult(result, basePath, opName)
    % Save processed image to file
    cpuResult = gather(result);

    % Ensure proper path formatting
    basePath = convertStringsToChars(basePath);
    if iscell(basePath)
        basePath = basePath{1};
    end

    % Split base path into components
    [fpath, fname, fext] = fileparts(basePath);

    % Default to PNG format if no extension
    if isempty(fext)
        fext = '.png';  % Set default extension
    end

    % Construct valid output path
    newFilename = [fname '-' opName fext];
    if isempty(fpath)
        outputPath = newFilename;
    else
        outputPath = fullfile(fpath, newFilename);
    end

    % Ensure output directory exists
    if ~isempty(fpath) && ~exist(fpath, 'dir')
        mkdir(fpath);
    end

    % Verify image data type
    if ~isfloat(cpuResult)
        cpuResult = im2double(cpuResult);
    end

    imwrite(cpuResult, outputPath);
end

% Processing operations
function y = copyOperation(x), y = x; end
function y = inversionOperation(x), y = imcomplement(x); end
function y = grayscaleOperation(x)
    y = cat(3, rgb2gray(x), rgb2gray(x), rgb2gray(x));
end
function y = thresholdOperation(x)
    y = uint8(x > 127) * 255;
end
function y = erodeOperation(x, se), y = imerode(x, se); end
function y = dilateOperation(x, se), y = imdilate(x, se); end
function y = convOperation(x, k), y = imfilter(x, k, 'conv'); end
function y = gaussOperation(x), y = imgaussfilt(x, 0.5, 'FilterSize', 3); end
