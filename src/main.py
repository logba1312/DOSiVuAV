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

def findFile(filename, search_path):
    # Iterate over all files matching the filename in the search_path directory and its subdirectories
    for file in search_path.rglob(filename):
        return file  # Return the first matching file
    return None  # Return None if no file is found

def main():
    # _, mtx, dist, rvecs, tvecs = cc.calibrateCamera()
    # np.savez(r'C:\Users\david\OneDrive\Documents\DOSiVuAV\Zadatak\camera_cal\calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # Load one of the test images
    img = cv2.imread(findFile('challange00111.jpg', Path(__file__).parent.parent))

    if img.shape[1] != 1280 or img.shape[0] != 720:
        # Resize the image to specific dimensions
        newWidth, newHeight = 1280, 720
        resizedImage = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)
        img = resizedImage

    undistortedImage = undistort.undistort(img)
    warpedImg = wi.warpImage(undistortedImage)
    binaryImage = bi.binaryImage(warpedImg)
    laneCooridnates = hist.histogramWithPeaks(binaryImage)

if __name__ == '__main__':
    main()