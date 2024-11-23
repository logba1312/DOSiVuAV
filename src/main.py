import numpy as np
import cv2 as cv
import glob

import cameraCalibration as cc
import undistort
import warpImage as wi

def main():
    _, mtx, dist, rvecs, tvecs = cc.calibrateCamera()
    np.savez(r'C:\Users\david\OneDrive\Documents\DOSiVuAV\Zadatak\camera_cal\calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    img = undistort.undistort()
    wi.warpImage(img)

if __name__ == '__main__':
    main()