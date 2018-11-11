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
    # reading in the image
    image_name = '0.jpg'
    image = cv2.imread('images/'+image_name)
    # mask away the text on the bottom of the image
    ymax = 650
    xmax = 1280
    mask = np.zeros(image.shape[:2],np.uint8) #720x1280
    mask[0:ymax,0:xmax] = 255
    image = cv2.bitwise_and(image,image,mask = mask)

    # color threshold to extract a range of white colors only that is associated with the lines on the ground
    low_threshold = 120
    high_threshold = 200
    lower_white = np.array([low_threshold,low_threshold,low_threshold])
    upper_white = np.array([high_threshold,high_threshold,high_threshold])
    mask = cv2.inRange(image,lower_white,upper_white)
    masked_image = cv2.bitwise_and(image,image,mask=mask)
    # converting the masked image to gray scale
    grayImage = grayscale(masked_image)

    # applying guassian blur to the grayScale Image
    kernel_size = 5
    blurredImage = gaussianBlur(grayImage,kernel_size)

    # applying canny edge detection (using automatic mean value thresholding)
    median_value = np.median(blurredImage)
    sigma = 0.33
    low_threshold = int(max(0, (1 - sigma) * median_value))
    high_threshold = int(min(255, (1 + sigma) * median_value))
    edges = canny(blurredImage, low_threshold, high_threshold)

    # performing hough transform
    theta = np.pi / 180
    rho = 1
    threshold = 40
    minLen = 50
    maxGap = 20
    houghImage = houghLines(edges, rho, theta, threshold, minLen, maxGap)

    # dilate to make the lines stronger
    dilated = cv2.dilate(houghImage, None, iterations=2)

    imageToSave = cv2.cvtColor(dilated, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_images/'+image_name,imageToSave)
    while True:
        # for debugging
        cv2.imshow("result", imageToSave)
        #cv2.imshow("blurredImage", blurredImage)
        #cv2.imshow("dilated", dilated)
        #cv2.imshow("res", image)
        #cv2.imshow("canny", edges)
        #cv2.imshow("hough", houghImage)

        # if the `q` key is pressed, break from the lop
        key = cv2.waitKey(1)
        if key == ord("q"):
            break 
    
    cv2.destroyAllWindows()


