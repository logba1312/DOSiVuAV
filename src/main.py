import numpy as np
import cv2 
import glob
import matplotlib.pyplot as plt
from pathlib import Path


import cameraCalibration as cc
import undistort
import warpImage as wi
import binaryImage as bi
import histogramWithPeaks as hist
import detectLanes as dl

def findFile(filename, search_path):
    # Iterate over all files matching the filename in the search_path directory and its subdirectories
    for file in search_path.rglob(filename):
        return file  # Return the first matching file
    return None  # Return None if no file is found

def main():
    # _, mtx, dist, rvecs, tvecs = cc.calibrateCamera()
    # np.savez(r'C:\Users\david\OneDrive\Documents\DOSiVuAV\Zadatak\camera_cal\calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # Load one of the test images
    img = cv2.imread(findFile('test2.jpg', Path(__file__).parent.parent))

    # Change image size if the resolusion is different from camera resolution
    if img.shape[1] != 1280 or img.shape[0] != 720:
        # Resize the image to specific dimensions
        newWidth, newHeight = 1280, 720
        resizedImage = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)
        img = resizedImage

    undistortedImage = undistort.undistort(img)
    warpedImg = wi.warpImage(undistortedImage)
    binaryImage, sobelImage = bi.binaryImage(warpedImg)
    laneCooridnates = hist.histogramWithPeaks(binaryImage)
    leftLane = 0
    rightLane = 0

    for point in laneCooridnates:
        if point <= 650:
            leftLane = point
        else:
            rightLane = point
            break

    vehiclePosition = (675 - leftLane) * 0.006 # vehicle offset from center in meters

    # Plot the histogram
    plt.figure()
    plt.imshow(warpedImg)
    plt.scatter([leftLane, rightLane], [300, 300], color='red', label=['Left', 'Right'])
    plt.scatter([leftLane, rightLane], [700, 700], color='red', label=['Left', 'Right'])
    plt.plot([leftLane, leftLane], [300, 720], color='red', linewidth=2)
    plt.plot([rightLane, rightLane], [300, 720], color='red', linewidth=2)
    plt.title('Location of left and right lane')
    plt.text(5, 35, f"Vehicle offset from center: {vehiclePosition:.4f}m", fontsize=12, color='yellow')
    plt.show()

    filteredLanes = dl.detectVerticalLines(binaryImage, leftLane, rightLane)

    # Visualize the result
    for x1, y1, x2, y2 in filteredLanes:
        cv2.line(warpedImg, (x1, y1), (x2, y2), (0, 255, 0), 5)

    plt.imshow(cv2.cvtColor(warpedImg, cv2.COLOR_BGR2RGB))
    plt.title("Detected Vertical Lines")
    plt.text(5, 35, f"Vehicle offset from center: {vehiclePosition:.4f}m", fontsize=12, color='white')
    plt.show()

if __name__ == '__main__':
    main()