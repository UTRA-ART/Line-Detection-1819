import numpy as np
import cv2


if __name__ == '__main__':
    image = cv2.imread('images/0.jpg')
    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    cv2.imshow('image', gray_image)
    cv2.waitKey(0)
