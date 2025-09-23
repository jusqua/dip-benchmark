# Digital Image Processing Benchmark

Digital image processing benchmark using multiple tools to learn new ways to develop image processors.

## Final results

Results from commit 4dbe28731e412793e406f43b36e0ee356ea2b635.

### Testing image

![Eye with diabetic retinopathy](assets/fundus.jpg)

You can find this image as **dr_15.JPG** zipped in **images of eyes with diabetic retinopathy** hosted at [High-Resolution Fundus (HRF) Image Database](https://www5.cs.fau.de/research/data/fundus-images/).

### Testing environment

#### Operating system

**Fedora Linux**

|         | Version      |
|---------|:------------:|
| Fedora  | 42 (Adams)   |
| Kernel  | 6.16.7-200   |

#### Device

**NVIDIA GeForce RTX 5060 Ti**

|         | Version      |
|---------|:------------:|
| Driver  | 580.82.07    |
| CUDA    | 13.0.84      |
| OpenCL  | 3.0          |

### Benchmark results

![Memory operations plot](assets/memory-operations.png)
![Point operations plot](assets/point-operations.png)
![Kernel passing operations plot](assets/kernel-operations.png)

#### [Julia + CUDA.jl](cuda/)

|         | Version      |
|---------|:------------:|
| Julia   | 1.11.7       |
| CUDA    | CUDA 13.0.84 |
| CUDA.jl | 5.8.3        |

| Operations                                 | Once               | Repeatedly                |
|--------------------------------------------|-------------------:|--------------------------:|
| Upload                                     |    0.00577s (once) |  53.791846s (10000 times) |
| Download                                   |   0.005981s (once) |  59.254799s (10000 times) |
| Copy                                       |   0.001923s (once) |   4.948093s (10000 times) |
| Inversion                                  |   0.061417s (once) |   5.209675s (10000 times) |
| Grayscale                                  |   0.088535s (once) |   5.712192s (10000 times) |
| Threshold                                  |   0.079148s (once) |   5.219541s (10000 times) |
| Erosion (3x3 Cross Kernel)                 |   0.100149s (once) |  16.582907s (10000 times) |
| Erosion (3x3 Square Kernel)                |     3.6e-5s (once) |  18.320976s (10000 times) |
| Erosion (1x3+3x1 Square Kernel)            |     3.9e-5s (once) |  22.972368s (10000 times) |
| Dilation (3x3 Cross Kernel)                |    0.07428s (once) |   16.87779s (10000 times) |
| Dilation (3x3 Square Kernel)               |     3.9e-5s (once) |   18.98323s (10000 times) |
| Dilation (1x3+3x1 Square Kernel)           |     4.3e-5s (once) |  24.373622s (10000 times) |
| Convolution (3x3 Gaussian Blur Kernel)     |   0.068092s (once) |  14.505329s (10000 times) |
| Convolution (1x3+3x1 Gaussian Blur Kernel) |     3.9e-5s (once) |  18.775434s (10000 times) |
| Convolution (5x5 Gaussian Blur Kernel)     |     3.8e-5s (once) |  27.490979s (10000 times) |
| Convolution (1x5+5x1 Gaussian Blur Kernel) |     4.0e-5s (once) |  22.585139s (10000 times) |
| Gaussian Blur (3x3 Kernel)                 |   0.127516s (once) |  18.824935s (10000 times) |

#### [Python + OpenCV with UMat (OpenCL backend)](opencv/)

|        | Version                 |
|--------|:-----------------------:|
| Python | 3.13.7                  |
| OpenCL | OpenCL 3.0 CUDA 13.0.84 |
| OpenCV | 3.13.7                  |

| Operations                                 | Once               | Repeatedly                |
|--------------------------------------------|-------------------:|--------------------------:|
| Upload                                     |   0.001169s (once) |  11.625007s (10000 times) |
| Download                                   |   0.003070s (once) |  15.275250s (10000 times) |
| Copy                                       |   0.001478s (once) |   2.939967s (10000 times) |
| Inversion                                  |   0.000173s (once) |   1.255894s (10000 times) |
| Grayscale                                  |   0.000681s (once) |   2.214707s (10000 times) |
| Threshold                                  |   0.000153s (once) |   1.251392s (10000 times) |
| Erosion (3x3 Cross Kernel)                 |   0.000615s (once) |   2.586656s (10000 times) |
| Erosion (3x3 Square Kernel)                |   0.000466s (once) |   2.702258s (10000 times) |
| Erosion (1x3+3x1 Square Kernel)            |   0.000784s (once) |   5.056510s (10000 times) |
| Dilation (3x3 Cross Kernel)                |   0.000499s (once) |   2.592111s (10000 times) |
| Dilation (3x3 Square Kernel)               |   0.000465s (once) |   2.702306s (10000 times) |
| Dilation (1x3+3x1 Square Kernel)           |   0.000787s (once) |   5.057605s (10000 times) |
| Convolution (3x3 Gaussian Blur Kernel)     |   0.000533s (once) |   2.327287s (10000 times) |
| Convolution (1x3+3x1 Gaussian Blur Kernel) |   0.000791s (once) |   4.289730s (10000 times) |
| Convolution (5x5 Gaussian Blur Kernel)     |   0.000600s (once) |   3.275498s (10000 times) |
| Convolution (1x5+5x1 Gaussian Blur Kernel) |   0.000904s (once) |   5.057216s (10000 times) |
| Gaussian Blur (3x3 Kernel)                 |   0.003612s (once) |  39.398873s (10000 times) |

#### [MATLAB](matlab/)

|        | Version                 |
|--------|:-----------------------:|
| CUDA   | CUDA 13.0.84            |
| MATLAB | 25.1.0.2943329 (R2025a) |

| Operations                                 | Once               | Repeatedly                |
|--------------------------------------------|-------------------:|--------------------------:|
| Upload                                     |   0.003165s (once) |  10.681811s (10000 times) |
| Download                                   |   0.007737s (once) |  19.213693s (10000 times) |
| Copy                                       |   0.000741s (once) |   0.001422s (10000 times) |
| Inversion                                  |   0.004338s (once) |   0.078044s (10000 times) |
| Grayscale                                  |   0.074586s (once) |   6.655125s (10000 times) |
| Threshold                                  |   0.022826s (once) |   0.016374s (10000 times) |
| Erosion (3x3 Cross Kernel)                 |   0.026478s (once) |   8.758027s (10000 times) |
| Erosion (3x3 Square Kernel)                |   0.001424s (once) |   9.313341s (10000 times) |
| Erosion (1x3+3x1 Square Kernel)            |   0.004469s (once) |  11.150636s (10000 times) |
| Dilation (3x3 Cross Kernel)                |   0.014315s (once) |   8.657920s (10000 times) |
| Dilation (3x3 Square Kernel)               |   0.001507s (once) |   9.201732s (10000 times) |
| Dilation (1x3+3x1 Square Kernel)           |   0.005421s (once) |  11.035648s (10000 times) |
| Convolution (3x3 Gaussian Blur Kernel)     |   0.031324s (once) |  23.913314s (10000 times) |
| Convolution (1x3+3x1 Gaussian Blur Kernel) |   0.060762s (once) |  52.888945s (10000 times) |
| Convolution (5x5 Gaussian Blur Kernel)     |   0.002801s (once) |  24.078633s (10000 times) |
| Convolution (1x5+5x1 Gaussian Blur Kernel) |   0.005824s (once) |  52.763706s (10000 times) |
| Gaussian Blur (3x3 Kernel)                 |   0.018833s (once) |  20.477430s (10000 times) |

#### [SYCL with AdaptiveCPP (CUDA backend)](sycl/)

|             | Version                                          |
|-------------|:------------------------------------------------:|
| Clang       | 20.1.8 (Fedora 20.1.8-4.fc42)                    |
| CUDA        | CUDA 13.0.84                                     |
| AdaptiveCPP | develop fe13c2d1f1244d9a2faffeee0bb34e6bfec18aba |

| Operations                                 | Once               | Repeatedly                |
|--------------------------------------------|-------------------:|--------------------------:|
| Copy (Host to Device)                      |   0.001493s (once) |  11.094086s (10000 times) |
| Copy (Device to Host)                      |   0.001604s (once) |  15.889138s (10000 times) |
| Copy (Device to Device)                    |   0.000181s (once) |   1.339829s (10000 times) |
| Invertion                                  |   0.056354s (once) |   1.862045s (10000 times) |
| Grayscale                                  |   0.042263s (once) |   1.616300s (10000 times) |
| Threshold                                  |   0.047320s (once) |   1.878173s (10000 times) |
| Erosion (3x3 Cross Kernel)                 |   0.054292s (once) |   8.444764s (10000 times) |
| Erosion (3x3 Square Kernel)                |   0.000995s (once) |   9.791751s (10000 times) |
| Erosion (1x3+3x1 Square Kernel)            |   0.000874s (once) |   8.724617s (10000 times) |
| Dilation (3x3 Cross Kernel)                |   0.056801s (once) |   8.450215s (10000 times) |
| Dilation (3x3 Square Kernel)               |   0.001001s (once) |   9.777083s (10000 times) |
| Dilation (1x3+3x1 Square Kernel)           |   0.000881s (once) |   8.542118s (10000 times) |
| Convolution (3x3 Gaussian Blur Kernel)     |   0.053168s (once) |   9.295212s (10000 times) |
| Convolution (1x3+3x1 Gaussian Blur Kernel) |   0.000869s (once) |   8.484524s (10000 times) |
| Convolution (5x5 Gaussian Blur Kernel)     |   0.001889s (once) |  19.320444s (10000 times) |
| Convolution (1x5+5x1 Gaussian Blur Kernel) |   0.001294s (once) |  11.692046s (10000 times) |
| Gaussian Blur (3x3 Kernel)                 |   0.058694s (once) |   3.115040s (10000 times) |

#### [VisionGL (OpenCL backend)](visiongl/)

|             | Version                                          |
|-------------|:------------------------------------------------:|
| Clang       | 20.1.8 (Fedora 20.1.8-4.fc42)                    |
| OpenCL      | OpenCL 3.0 CUDA 13.0.84                          |
| VisionGL    | develop 45078178be564f59c601fca295afca9897b90dba |

| Operations                                 | Once               | Repeatedly                |
|--------------------------------------------|-------------------:|--------------------------:|
| Copy (Host to Device)                      |   0.001963s (once) |  17.037199s (10000 times) |
| Copy (Device to Host)                      |   0.002078s (once) |  19.480001s (10000 times) |
| Copy (Device to Device)                    |   0.004619s (once) |   1.584068s (10000 times) |
| Invertion                                  |   0.000936s (once) |   1.584519s (10000 times) |
| Grayscale                                  |   0.000839s (once) |   1.584666s (10000 times) |
| Threshold                                  |   0.001212s (once) |   1.584724s (10000 times) |
| Erosion (3x3 Cross Kernel)                 |   0.001558s (once) |   3.192812s (10000 times) |
| Erosion (3x3 Square Kernel)                |   0.000432s (once) |   3.143420s (10000 times) |
| Erosion (1x3+3x1 Square Separated Kernel)  |   0.000652s (once) |   5.047741s (10000 times) |
| Dilation (3x3 Cross Kernel)                |   0.001610s (once) |   3.191952s (10000 times) |
| Dilation (3x3 Square Kernel)               |   0.000433s (once) |   3.200868s (10000 times) |
| Dilation (1x3+3x1 Square Kernel)           |   0.000613s (once) |   5.067978s (10000 times) |
| Convolution (3x3 Gaussian Blur Kernel)     |   0.001613s (once) |   3.024686s (10000 times) |
| Convolution (1x3+3x1 Gaussian Blur Kernel) |   0.000603s (once) |   5.011536s (10000 times) |
| Convolution (5x5 Gaussian Blur Kernel)     |   0.000540s (once) |   4.675456s (10000 times) |
| Convolution (1x5+5x1 Gaussian Blur Kernel) |   0.000667s (once) |   5.557838s (10000 times) |
| Gaussian Blur (3x3 Kernel)                 |   0.001445s (once) |   1.934583s (10000 times) |
