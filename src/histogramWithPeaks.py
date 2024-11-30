import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def histogramWithPeaks(image):
    image = image // 255

    # Check if the image is binary (contains only 0 and 1)
    if not np.all(np.isin(image, [0, 1])):
        raise ValueError("The input image should be binary (only 0s and 1s).")

    # Sum the number of pixels with value 1 in each column
    histogram = np.sum(image, axis=0)  # axis=0 sums along the columns

    # Find peaks in the histogram
    peaks, _ = find_peaks(histogram, height=100, distance=50)

    print(peaks)
    # print(histogram[peaks])

    # Plot the histogram
    # plt.figure(figsize=(10, 6))
    # plt.plot(histogram)
    # plt.scatter(peaks, histogram[peaks], color='red', label='Peaks')
    # plt.title('Histogram of Pixel Intensity in Each Column')
    # plt.xlabel('Column Index')
    # plt.ylabel('Number of 1 Pixels')
    # plt.grid(True)
    # plt.show()

    return peaks