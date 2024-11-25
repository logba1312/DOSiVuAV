import numpy as np
import cv2
import glob
from pathlib import Path

def findFile(filename, search_path):
    # Iterate over all files matching the filename in the search_path directory and its subdirectories
    for file in search_path.rglob(filename):
        return file  # Return the first matching file
    return None  # Return None if no file is found

def undistort(img):
    # Load calibrated camera parameters
    calibratio = np.load(findFile('calib.npz', Path(__file__).parent.parent))
    mtx = calibratio['mtx']
    dist = calibratio['dist']
    rvecs = calibratio['rvecs']
    tvecs = calibratio['tvecs']
    
    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    # Display the final result
    cv2.imshow('chess board', np.hstack((img, undistortedImg)))
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return undistortedImg
