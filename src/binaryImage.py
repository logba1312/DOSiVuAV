import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def binaryImage(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv2.imshow("Blurred", blurred)

    # Compute gradients
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x
    sobelY = 0 # cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y

    gradient = np.sqrt(sobelX**2 + sobelY**2)
    gradient = np.uint8(255 * gradient / np.max(gradient))

    # cv2.imshow("Sobel", gradient)
    # cv2.waitKey(0)

    _, sobelBinary = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Threshold 50", sobelBinary)
    # cv2.waitKey(0)

    # Create a binary mask for yellow and white colors in the HLS space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lowerYellow = np.array([18, 130, 100])
    upperYellow = np.array([30, 255, 255])
    yellowMask = cv2.inRange(hls, lowerYellow, upperYellow)

    lowerWhite = np.array([0, 200, 0])
    upperWhite = np.array([255, 255, 255])
    whiteMask = cv2.inRange(hls, lowerWhite, upperWhite)

    # plt.imshow(hls)
    # plt.show()

    # Combine the yellow and white masks
    colorMask = cv2.bitwise_or(yellowMask, whiteMask)
    # cv2.imshow("Color mask", colorMask)
    # cv2.waitKey(0)

    # Combine color and edge detection results
    combined = cv2.bitwise_or(sobelBinary, colorMask)
    # cv2.imshow("Sobel + color mask", combined)
    # cv2.waitKey(0)

    # Morphological closing to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Morphological", closed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return closed, gradient