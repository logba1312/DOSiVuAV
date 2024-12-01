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
from fitPolynomial import fitPolynomials, calculateCurvature

def findFile(filename, search_path):
    # Iterate over all files matching the filename in the search_path directory and its subdirectories
    for file in search_path.rglob(filename):
        return file  # Return the first matching file
    return None  # Return None if no file is found

def main():
    # _, mtx, dist, rvecs, tvecs = cc.calibrateCamera()
    # np.savez(r'C:\Users\david\OneDrive\Documents\DOSiVuAV\Zadatak\camera_cal\calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Load one of the test images/videos
    # fileName = 'test6.jpg'
    # displayImageResult(fileName)
    fileName = 'challenge01.mp4'
    displayVideoResult(fileName)

def laneDetection(img):
    # Change image size if the resolusion is different from camera resolution
    if img.shape[1] != 1280 or img.shape[0] != 720:
        # Resize the image to specific dimensions
        newWidth, newHeight = 1280, 720
        resizedImage = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)
        img = resizedImage

    # Process the image
    undistortedImage = undistort.undistort(img)
    warpedImg, warpMatrix = wi.warpImage(undistortedImage)
    binaryImage = bi.binaryImage(warpedImg)
    laneCoordinates = hist.histogramWithPeaks(binaryImage)
    
    # Calculate the x coordinates of the lef and right lane and the vehicle position
    leftLaneLine, rightLaneLine, vehiclePosition = calculateLanesAndVehiclePosition(laneCoordinates)

    # Obtain the lines found on the position of left and right lanes
    filteredLines = dl.detectVerticalLines(binaryImage, leftLaneLine, rightLaneLine)

    # Fit a polynomial through each lane line
    (leftXVals, leftYVals, leftCoeff), (rightXVals, rightYVals, rightCoeff) = fitPolynomials(filteredLines, warpedImg.shape[0], leftLaneLine, rightLaneLine)

    # Visualization
    polyImage = warpedImg.copy()

    # Draw the left polynomial
    if leftXVals is not None and leftYVals is not None:
        for i in range(len(leftYVals) - 1):
            cv2.line(polyImage, (int(leftXVals[i]), int(leftYVals[i])), (int(leftXVals[i+1]), int(leftYVals[i+1])), (255, 0, 0), 5)

    # Draw the right polynomial
    if rightXVals is not None and rightYVals is not None:
        for i in range(len(rightYVals) - 1):
            cv2.line(polyImage, (int(rightXVals[i]), int(rightYVals[i])), (int(rightXVals[i+1]), int(rightYVals[i+1])), (0, 0, 255), 5)

    # Calculate the curvature of the lanes
    leftCurvature = calculateCurvature(leftCoeff, polyImage.shape[0])
    rightCurvature = calculateCurvature(rightCoeff, polyImage.shape[0])

    # Display the result of fitPolinomials
    cv2.putText(
        polyImage,
        f"Left line curvature: {leftCurvature:.2f} m",  # Text content
        (5, 100),  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        1,  # Font scale
        (255, 0, 0),  # Font color (Blue)
        2,  # Thickness
        cv2.LINE_AA,  # Line type
    )
    cv2.putText(
        polyImage,
        f"Right line curvature: {rightCurvature:.2f} m",  # Text content
        (5, 150),  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        1,  # Font scale
        (0, 0, 255),  # Font color (Red)
        2,  # Thickness
        cv2.LINE_AA,  # Line type
    )
    cv2.putText(
        polyImage,
        f"Vehicle offset: {vehiclePosition:.2f} m",  # Text content
        (5, 200),  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        1,  # Font scale
        (255, 255, 0),  # Font color (Light blue)
        2,  # Thickness
        cv2.LINE_AA,  # Line type
    )

    polyImage = fillLaneTransparent(polyImage, leftXVals, leftYVals, rightXVals, rightYVals)
    cv2.imshow("Fitted Polynomial", polyImage) 
    # cv2.imwrite("./output/polyImage.jpg", polyImage)

    # outputImage = overlayLaneOnOriginal(img, polyImage, warpMatrix)
    # cv2.imwrite("./output/unwarpedImage.jpg", outputImage)
    # cv2.imshow("Final", outputImage)

    return polyImage

    
def displayImageResult(imgName):
    # Open image
    img = cv2.imread(findFile(imgName, Path(__file__).parent.parent))

    # Run lane detection algorithm
    resultImage = laneDetection(img)
    cv2.waitKey(0)

def displayVideoResult(videoName):
    # Path to the video file
    video_capture = cv2.VideoCapture(findFile(videoName, Path(__file__).parent.parent))
    out = cv2.VideoWriter("./output/finalVideo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))

    # Check if the video file was opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Process video frames
    while True:
        ret, frame = video_capture.read()  # Read a frame from the video

        # Break the loop if no frame is returned (end of video)
        if not ret:
            break

        # Run lane detection algorithm
        modifiedFrame = laneDetection(frame)
        out.write(modifiedFrame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

def calculateLanesAndVehiclePosition(laneCoordinates):
    # Calculate the x coordinates of the lef and right lane (closest two lines to the center of the image, one lef one right)
    for point in laneCoordinates:
        if point <= 665:            # 665 aproximatley the middle of the car in the warped image
            leftLane = point
        else:
            rightLane = point
            break

    vehiclePosition = ((leftLane + rightLane) / 2 - 665) * 0.016  # Vehicle offset from center in meters
    # 0.016 is calculated by dividing the number of pixels between the twto lanes, with the average lane width of USA highway lanes which is 3.7 meters

    return leftLane, rightLane, vehiclePosition

def fillLaneTransparent(image, leftXVals, leftYVals, rightXVals, rightYVals):
    # Create an empty mask image
    laneMask = np.zeros_like(image, dtype=np.uint8)

    # Combine the left and right lane points into a single polygon
    leftLanePoints = np.array(np.flipud(np.column_stack((leftXVals, leftYVals))))  # Flip vertically
    rightLanePoints = np.array(np.column_stack((rightXVals, rightYVals)))  # Stack as (x, y)
    lanePolygon = np.vstack((leftLanePoints, rightLanePoints)).astype(np.int32)  # Combine left and right

    # Fill the polygon on the mask with green
    cv2.fillPoly(laneMask, [lanePolygon], (0, 255, 0))  # Green color

    # Create a weighted overlay for transparency
    transparentLane = cv2.addWeighted(image, 1, laneMask, 0.3, 0)  # Adjust 0.3 for more/less transparency

    return transparentLane

def overlayLaneOnOriginal(originalImage, polyImage, warpMatrix):
    # Compute the inverse perspective transform
    inverseWarpMatrix = np.linalg.inv(warpMatrix)

    # Warp the filled lane image back to the original perspective
    originalPerspectiveLane = cv2.warpPerspective(polyImage, inverseWarpMatrix, (originalImage.shape[1], originalImage.shape[0]))

    # Combine the lane mask with the original image
    overlayImage = cv2.addWeighted(originalImage, 1, originalPerspectiveLane, 0.3, 0)  # Adjust transparency (0.3)
    
    return overlayImage

if __name__ == '__main__':
    main()