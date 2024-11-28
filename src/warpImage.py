import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def selectPoints(event, x, y, flags, param):
    # If button is clicked, print the poin coordinates in terminal
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point selected: {x, y}")

def warpImage(image):
    # cv2.imshow("Select Points", image)
    # cv2.setMouseCallback("Select Points", selectPoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Define the source points (corners of the region to warp)
    sourcePoints = np.float32([
        [600, 455],      # Top-left corner
        [718, 455],      # Top-right corner
        [350, 621],      # Bottom-left corner
        [984, 621]      # Bottom-right corner
    ])

    # Define the destination points (top-down rectangle)
    destinationPoints = np.float32([
        [350 + 200, 0],         # Top-left corner
        [984 - 200, 0],         # Top-right corner
        [350 + 200, 720],         # Bottom-left corner
        [984 - 200, 720]          # Bottom-right corner
    ])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(sourcePoints, destinationPoints)

    height, width = image.shape[:2]

    # Apply the perspective warp
    warpedImage = cv2.warpPerspective(image, matrix, (width, height))

    # Display the result
    # cv2.imshow("Warped Image", warpedImage)
    # plt.imshow(warpedImage)
    # cv2.imwrite("./output/warpedImage.jpg", warpedImage)
    # cv2.destroyAllWindows()

    return warpedImage, matrix