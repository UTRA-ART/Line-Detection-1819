import numpy as np
import cv2

# helper functions
def grayscale(img):
    '''Applies the grayscale Transform
    This will return an image with only one color channel
    to see the returned image as grayscale call plt.imshow(gray,cmap='gray')
    where gray is the returned image from this function'''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussianBlur(img, kernel_size):
    '''Applies a guassian noise kernel'''
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    '''Applies the canny transform'''
    return cv2.Canny(img, low_threshold, high_threshold)


def drawLines(image, lines, color=[255, 0, 0], thickness=2):
    '''draws the lines given in lines argument on the image'''
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def houghLines(image, rho, theta, threshold, minLength, maxGap):
    '''image should be the output of a canny transform
    returns an image with hough lines drawn'''
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=minLength, maxLineGap=maxGap)
    lineImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    drawLines(lineImage, lines)
    return lineImage


if __name__ == '__main__':
    image = cv2.imread('images/0.jpg')
    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    cv2.imshow('image', gray_image)
    cv2.waitKey(0)
