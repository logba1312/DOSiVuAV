import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectVerticalLines(image, leftLaneCoordinate, rightLaneCoordinate, tolerance=20):
    # Hough Line Transform
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=30, minLineLength=5, maxLineGap=200)

    filtered_lines = []

    xRange = (leftLaneCoordinate, rightLaneCoordinate)

    # Filter for vertical lines around specified x-coordinates
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 30:  # Check if the line is vertical
                x_coord = (x1 + x2) // 2  # Average x-coordinate
                if any(abs(x_coord - x) <= tolerance for x in xRange):
                    filtered_lines.append((x1, y1, x2, y2))

    return filtered_lines

# # Example usage
# image = cv2.imread("your_image_path.jpg")  # Replace with your image path
# xRange = (600, 800)  # Target x-coordinates
# lines = detectVerticalLines(image, xRange)

# # Visualize the result
# for x1, y1, x2, y2 in lines:
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Detected Vertical Lines")
# plt.show()
