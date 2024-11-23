import numpy as np
import cv2
import glob
import os

def calibrateCamera():
    # Define the chess board rows and columns
    rows = 6
    columns = 9

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
    objectPoints = np.zeros(shape=(rows * columns, 3), dtype=np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

    # Create the arrays to store the object points and the image points
    objectPointsArray = []
    imagePointsArray = []

    for path in glob.glob(r'C:\Users\david\OneDrive\Documents\DOSiVuAV\Zadatak\camera_cal/calibration*.jpg'):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
        # cv2.imshow('image', img)

        # Make sure the chess board pattern was found in the image
        if ret:
            print("Pattern found!")
            # Refine the corner position
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Add the object points and the image points to the arrays
            objectPointsArray.append(objectPoints)
            imagePointsArray.append(corners)

            # Draw the corners on the image
            cv2.drawChessboardCorners(img, (rows, columns), corners, ret)

        # Display the image
        cv2.imshow('chess board', img)
        # cv2.waitKey(0)

    return cv2.calibrateCamera(objectPointsArray, imagePointsArray, gray.shape[::-1], None, None)