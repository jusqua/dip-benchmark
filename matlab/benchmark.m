function benchmark(infile, outdir, rounds)
    % Force forward compatibility for CUDA
    parallel.gpu.enableCUDAForwardCompatibility(true);

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
    aux = gpuArray(zeros(size(img), 'like', img));
    sample = gpuArray(zeros(size(img), 'like', img));
    [~, name, ext] = fileparts(filename);
    outputBase = fullfile(outdir, [name ext]);

    % Define processing kernels
    [cross, square, square_sep_1x3, square_sep_3x1, blur3, blur5, blur3_1x3, blur3_3x1, blur5_1x5, blur5_5x1] = createKernels();

    % Define operations with descriptions matching Python
    operations = {
        {@() copyOperation(gpuImg), 'Copy', 'copy'};
        {@() inversionOperation(gpuImg), 'Inversion', 'inversion'};
        {@() grayscaleOperation(gpuImg), 'Grayscale', 'grayscale'};
        {@() thresholdOperation(gpuImg), 'Threshold', 'threshold'};
        {@() erodeOperation(gpuImg, cross), 'Erosion (3x3 Cross Kernel)', 'erosion-cross'};
        {@() erodeOperation(gpuImg, square), 'Erosion (3x3 Square Kernel)', 'erosion-square'};
        {@() erodeSeparatedOperation(gpuImg, square_sep_1x3, square_sep_3x1, aux), 'Erosion (1x3+3x1 Square Kernel)', 'erosion-square-separated'};
        {@() dilateOperation(gpuImg, cross), 'Dilation (3x3 Cross Kernel)', 'dilation-cross'};
        {@() dilateOperation(gpuImg, square), 'Dilation (3x3 Square Kernel)', 'dilation-square'};
        {@() dilateSeparatedOperation(gpuImg, square_sep_1x3, square_sep_3x1, aux), 'Dilation (1x3+3x1 Square Kernel)', 'dilation-square-separated'};
        {@() convOperation(gpuImg, blur3), 'Convolution (3x3 Gaussian Blur Kernel)', 'convolution-gaussian-blur-3x3'};
        {@() convSeparatedOperation(gpuImg, blur3_1x3, blur3_3x1, aux), 'Convolution (1x3+3x1 Gaussian Blur Kernel)', 'convolution-gaussian-blur-3x3-separated'};
        {@() convOperation(gpuImg, blur5), 'Convolution (5x5 Gaussian Blur Kernel)', 'convolution-gaussian-blur-5x5'};
        {@() convSeparatedOperation(gpuImg, blur5_1x5, blur5_5x1, aux), 'Convolution (1x5+5x1 Gaussian Blur Kernel)', 'convolution-gaussian-blur-5x5-separated'};
        {@() gaussOperation(gpuImg), 'Gaussian Blur (3x3 Kernel)', 'gaussian-blur-3x3'};
    };

    % Find the longest description for formatting
    maxDescLength = 0;
    for i = 1:size(operations, 1)
        descLength = length(operations{i}{2});
        if descLength > maxDescLength
            maxDescLength = descLength;
        end
    end

    % Benchmark pipeline
    for i = 1:size(operations, 1)
        procFunc = operations{i}{1};
        description = operations{i}{2};
        opName = operations{i}{3};

        [onceTime, totalTime] = timeOperation(procFunc, rounds);
        saveResult(procFunc(), outputBase, opName);

        % Format output to match Python exactly
        paddedDesc = sprintf('%-*s', maxDescLength, description);
        fprintf('| %s | %10.6fs (once) | %10.6fs (%d times) |\n', ...
                paddedDesc, onceTime, totalTime, rounds);
    end
end

function [cross, square, square_sep_1x3, square_sep_3x1, blur3, blur5, blur3_1x3, blur3_3x1, blur5_1x5, blur5_5x1] = createKernels()
    % Create processing kernels matching Python implementation
    cross = strel('arbitrary', [0 1 0; 1 1 1; 0 1 0]);
    square = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]);

    % Separated kernels for morphological operations
    square_sep_1x3 = strel('arbitrary', [1 1 1]);
    square_sep_3x1 = strel('arbitrary', [1; 1; 1]);

    % Gaussian blur kernels - double precision to match Python float32
    blur3 = gpuArray(double([1  2  1;
                            2  4  2;
                            1  2  1]/16));

    blur5 = gpuArray(double([1  4  6  4 1;
                            4 16 24 16 4;
                            6 24 36 24 6;
                            4 16 24 16 4;
                            1  4  6  4 1]/256));

    % Separated Gaussian blur kernels
    blur3_1x3 = gpuArray(double([1/4, 1/2, 1/4]));
    blur3_3x1 = gpuArray(double([1/4; 1/2; 1/4]));

    blur5_1x5 = gpuArray(double([1/16, 4/16, 6/16, 4/16, 1/16]));
    blur5_5x1 = gpuArray(double([1/16; 4/16; 6/16; 4/16; 1/16]));
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
    newFilename = [opName '-' fname fext];
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
    % Convert to grayscale and back to 3-channel to match Python behavior
    gray = rgb2gray(x);
    y = cat(3, gray, gray, gray);
end

function y = thresholdOperation(x)
    y = x >= 128;
end

function y = erodeOperation(x, se), y = imerode(x, se); end

function y = dilateOperation(x, se), y = imdilate(x, se); end

function y = erodeSeparatedOperation(x, se1, se2, aux)
    % Separated erosion: apply se1 then se2
    aux = imerode(x, se1);
    y = imerode(aux, se2);
end

function y = dilateSeparatedOperation(x, se1, se2, aux)
    % Separated dilation: apply se1 then se2
    aux = imdilate(x, se1);
    y = imdilate(aux, se2);
end

function y = convOperation(x, k), y = imfilter(x, k, 'conv'); end

function y = convSeparatedOperation(x, k1, k2, aux)
    % Separated convolution: apply k1 then k2
    aux = imfilter(x, k1, 'conv');
    y = imfilter(aux, k2, 'conv');
end

function y = gaussOperation(x), y = imgaussfilt(x, 0.5, 'FilterSize', 3); end
