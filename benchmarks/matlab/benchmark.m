function benchmark(infile, outdir, rounds)
    parallel.gpu.enableCUDAForwardCompatibility(true);

    if nargin < 3
        rounds = 10000;
    end

    try
        gpu = gpuDevice();
        fprintf('Using GPU: %s\n', gpu.Name);
    catch
        error('GPU device not available');
    end

    if ~exist(infile, 'file')
        error('Input file does not exist');
    end
    [img, filename] = readImage(infile);
    gpuImg = gpuArray(img);

    [~, name, ext] = fileparts(filename);
    outputBase = fullfile(outdir, [name ext]);

    [cross, square, square_sep_1x3, square_sep_3x1, blur3, blur5, blur3_1x3, blur3_3x1, blur5_1x5, blur5_5x1] = createKernels();
    operations = {
        {@() uploadOperation(img), 'Upload', ''};
        {@() downloadOperation(gpuImg), 'Download', ''};
        {@() copyOperation(gpuImg), 'Copy', 'copy'};
        {@() inversionOperation(gpuImg), 'Inversion', 'inversion'};
        {@() grayscaleOperation(gpuImg), 'Grayscale', 'grayscale'};
        {@() thresholdOperation(gpuImg), 'Threshold', 'threshold'};
        {@() erodeOperation(gpuImg, cross), 'Erosion (3x3 Cross Kernel)', 'erosion-cross'};
        {@() erodeOperation(gpuImg, square), 'Erosion (3x3 Square Kernel)', 'erosion-square'};
        {@() erodeSeparatedOperation(gpuImg, square_sep_1x3, square_sep_3x1), 'Erosion (1x3+3x1 Square Kernel)', 'erosion-square-separated'};
        {@() convOperation(gpuImg, blur3), 'Convolution (3x3 Gaussian Blur Kernel)', 'convolution-gaussian-blur-3x3'};
        {@() convSeparatedOperation(gpuImg, blur3_1x3, blur3_3x1), 'Convolution (1x3+3x1 Gaussian Blur Kernel)', 'convolution-gaussian-blur-3x3-separated'};
        {@() convOperation(gpuImg, blur5), 'Convolution (5x5 Gaussian Blur Kernel)', 'convolution-gaussian-blur-5x5'};
        {@() convSeparatedOperation(gpuImg, blur5_1x5, blur5_5x1), 'Convolution (1x5+5x1 Gaussian Blur Kernel)', 'convolution-gaussian-blur-5x5-separated'};
        {@() gaussOperation(gpuImg), 'Gaussian Blur (3x3 Kernel)', 'gaussian-blur-3x3'};
    };

    maxDescLength = 0;
    for i = 1:size(operations, 1)
        descLength = length(operations{i}{2});
        if descLength > maxDescLength
            maxDescLength = descLength;
        end
    end

    for i = 1:size(operations, 1)
        procFunc = operations{i}{1};
        description = operations{i}{2};
        opName = operations{i}{3};

        [onceTime, totalTime, result] = measureTime(procFunc, rounds);
        paddedDesc = sprintf('%-*s', maxDescLength, description);
        fprintf('| %s | %10.6fs (once) | %10.6fs (%d times) |\n', ...
            paddedDesc, onceTime, totalTime, rounds);

        if isempty(opName)
            continue
        end 

        saveResult(result, outputBase, opName);
    end
end

function [cross, square, square_sep_1x3, square_sep_3x1, blur3, blur5, blur3_1x3, blur3_3x1, blur5_1x5, blur5_5x1] = createKernels()
    cross = strel('arbitrary', [0 1 0; 1 1 1; 0 1 0]);
    square = strel('arbitrary', [1 1 1; 1 1 1; 1 1 1]);

    square_sep_1x3 = strel('arbitrary', [1 1 1]);
    square_sep_3x1 = strel('arbitrary', [1; 1; 1]);

    blur3 = gpuArray(double([1  2  1;
                            2  4  2;
                            1  2  1]/16));

    blur5 = gpuArray(double([1  4  6  4 1;
                            4 16 24 16 4;
                            6 24 36 24 6;
                            4 16 24 16 4;
                            1  4  6  4 1]/256));

    blur3_1x3 = gpuArray(double([1/4, 1/2, 1/4]));
    blur3_3x1 = gpuArray(double([1/4; 1/2; 1/4]));

    blur5_1x5 = gpuArray(double([1/16, 4/16, 6/16, 4/16, 1/16]));
    blur5_5x1 = gpuArray(double([1/16; 4/16; 6/16; 4/16; 1/16]));
end

function [img, filename] = readImage(path)
    img = imread(path);
    [~, name, ext] = fileparts(path);
    filename = [name ext];
end

function [onceTime, totalTime, result] = measureTime(operation, rounds)
    tic;
    result = operation();
    wait(gpuDevice);
    onceTime = toc;

    tic;
    for i = 1:rounds
        operation();
    end
    wait(gpuDevice);
    totalTime = toc / rounds;
end

function saveResult(result, basePath, opName)
    cpuResult = gather(result);

    basePath = convertStringsToChars(basePath);
    if iscell(basePath)
        basePath = basePath{1};
    end

    [fpath, fname, fext] = fileparts(basePath);

    if isempty(fext)
        fext = '.png';
    end

    newFilename = [opName '-' fname fext];
    if isempty(fpath)
        outputPath = newFilename;
    else
        outputPath = fullfile(fpath, newFilename);
    end

    if ~isempty(fpath) && ~exist(fpath, 'dir')
        mkdir(fpath);
    end

    if ~isfloat(cpuResult)
        cpuResult = im2double(cpuResult);
    end

    imwrite(cpuResult, outputPath);
end

function y = uploadOperation(x), y = gpuArray(x); end

function y = downloadOperation(x), y = gather(x); end

function y = copyOperation(x), y = gpuArray(x); end

function y = inversionOperation(x), y = imcomplement(x); end

function y = grayscaleOperation(x)
    gray = rgb2gray(x);
    y = cat(3, gray, gray, gray);
end

function y = thresholdOperation(x)
    y = x >= 128;
end

function y = erodeOperation(x, se), y = imerode(x, se); end

function y = erodeSeparatedOperation(x, se1, se2)
    y = imerode(imerode(x, se1), se2);
end

function y = convOperation(x, k), y = imfilter(x, k, 'conv'); end

function y = convSeparatedOperation(x, k1, k2)
    y = imfilter(imfilter(x, k1, 'conv'), k2, 'conv');
end

function y = gaussOperation(x), y = imgaussfilt(x, 0.5, 'FilterSize', 3); end
