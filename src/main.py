import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


import cameraCalibration as cc
import undistort
import warpImage as wi
import binaryImage as bi
import histogramWithPeaks as hist

def main():
    # _, mtx, dist, rvecs, tvecs = cc.calibrateCamera()
    # np.savez(r'C:\Users\david\OneDrive\Documents\DOSiVuAV\Zadatak\camera_cal\calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    img = undistort.undistort()
    warpedImg = wi.warpImage(img)
    binaryImage = bi.binaryImage(warpedImg)
    laneCooridnates = hist.histogramWithPeaks(binaryImage)

if __name__ == '__main__':
    main()