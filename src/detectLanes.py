import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectVerticalLines(image, leftLaneCoordinate, rightLaneCoordinate, tolerance=30):
    # Hough Line Transform
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=30, minLineLength=3, maxLineGap=100)

    filteredLines = []

    xRange = (leftLaneCoordinate, rightLaneCoordinate)

    # Filter for vertical lines around specified x-coordinates
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 30:  # Check if the line is vertical
                xCoord = (x1 + x2) // 2  # Average x-coordinate
                if any(abs(xCoord - x) <= tolerance for x in xRange):
                    filteredLines.append((x1, y1, x2, y2))

    return filteredLines