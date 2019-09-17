# ART Line Detection

This repository implements the vision pipeline for lane detection used in UTRA's autonomous rover team (ART).

## Dependencies
* Python 3.5
* OpenCV 3.0.0 or higher

# How to use this repository
All the code is contained within `main.py`. Running it will process the images in `input_images/` and return the images with the detected lanes annotated onto them, found in `test/`.

To ensure OpenCV is installed
1. Run `import cv2` in python3
2. It should return with no errors

Run main.py as `python3 main.py` on the bash terminal to execute line detection

# Implementation details
The lane detection is comprised of two parts:

## Image preprocessing
We first process input image in order to produce a binary image that only contains white lines, signifying the la on a black background. This is done by applying two different thresholding methods, on the Saturation Channel and Sobel thresholding, to two copies of the input image, respectively. The resulting binaries are combined to produce the resulting thresholded image, followed by a median blur to smooth out remaining white speckles. Finally, the binary image is transformed from perspective to bird's eye view before being passed to the lane detection module. See Results to see the generated binary.

## Lane detection
Lanes are detecting by performing a sliding-window based algorithm to the binary output from the preprocessing stage. First, a histogram is generates along a given axis based on the number of whtie pixels along each column. The histogram is then partitioned into bins along the axis, and bins whose histogram values exceed a heuristically determined cutoff are saved. For each of these bins, the coordinates of the pixels are obtained, and are passed into `np.plyfit`. Bucket sizes must be large enough to encompass highly curved images, but small enough so as not not overlap with a potentially neighbouring line.


# Results

1280x720 JPEG input from stereo camera: 
![alt inputimage](https://github.com/UTRA-ART/ART_Line_Detection/blob/master/images/3.jpg)

Image preprocessing:
![alt outputimage](https://github.com/UTRA-ART/ART_Line_Detection/blob/master/output_images/3.jpg)

Then using a sliding-window algorithm which polyfits in horizontal and vertical directions:
![alt finalimage](https://github.com/UTRA-ART/ART_Line_Detection/blob/master/final_images/3.png)
