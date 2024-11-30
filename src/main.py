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
from fitPolynomial import fitPolynomials

def findFile(filename, search_path):
    # Iterate over all files matching the filename in the search_path directory and its subdirectories
    for file in search_path.rglob(filename):
        return file  # Return the first matching file
    return None  # Return None if no file is found

def main():
    # _, mtx, dist, rvecs, tvecs = cc.calibrateCamera()
    # np.savez(r'C:\Users\david\OneDrive\Documents\DOSiVuAV\Zadatak\camera_cal\calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Load one of the test images/videos
    fileName = 'test6.jpg'
    displayImageResult(fileName)
    # fileName = 'challenge01.mp4'
    # displayVideoResult(fileName)

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
    (leftXVals, leftYVals), (rightXVals, rightYVals) = fitPolynomials(filteredLines, warpedImg.shape[0], leftLaneLine, rightLaneLine)

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

    # Display the result of fitPolinomials
    cv2.putText(
        polyImage,
        f"Vehicle offset: {vehiclePosition:.3f} m",  # Text content
        (5, 500),  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        1,  # Font scale
        (255, 255, 0),  # Font color (Light blue)
        2,  # Thickness
        cv2.LINE_AA,  # Line type
    )
    cv2.imshow("Fitted Polynomial", polyImage) 
    cv2.imwrite("./output/polyImage.jpg", polyImage)
    

    originalLines = warpToOriginal(filteredLines, warpMatrix)
    resultImage = drawLinesOnImage(img, originalLines)
    # cv2.imshow("Lines on Original Image", np.hstack((frame, displayImg, resultImage)))
    # cv2.imshow("Reverse warp", resultImage)
    # cv2.imwrite("./output/HoughLines.jpg", warpedImg)

    # Draw detected lanes on the warped image
    for x1, y1, x2, y2 in filteredLines:
        cv2.line(warpedImg, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Display the result 
    cv2.putText(
        warpedImg,
        f"Vehicle offset: {vehiclePosition:.3f} m",  # Text content
        (5, 500),  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        1,  # Font scale
        (255, 255, 0),  # Font color (Light blue)
        2,  # Thickness
        cv2.LINE_AA,  # Line type
    )
    cv2.imshow("Processed Frame", warpedImg)

    return resultImage

    
def displayImageResult(imgName):
    # Open image
    img = cv2.imread(findFile(imgName, Path(__file__).parent.parent))

    # Run lane detection algorithm
    resultImage = laneDetection(img)
    cv2.waitKey(0)

def displayVideoResult(videoName):
    # Path to the video file
    video_capture = cv2.VideoCapture(findFile(videoName, Path(__file__).parent.parent))
    # out = cv2.VideoWriter("finalVideo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280 * 3, 720))

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
        # out.write(np.hstack((frame, modifiedFrame)))

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    video_capture.release()
    # out.release()
    cv2.destroyAllWindows()

def calculateLanesAndVehiclePosition(laneCoordinates):
    # Calculate the x coordinates of the lef and right lane (closest two lines to the center of the image, one lef one right)
    for point in laneCoordinates:
        if point <= 665:            # 665 aproximatley the middle of the car in the warped image
            leftLane = point
        else:
            rightLane = point
            break

    vehiclePosition = ((leftLane + rightLane) / 2 - 665) * 0.006  # Vehicle offset from center in meters
    # 0.006 is calculated by dividing the number of pixels between the twto lanes, with the average lane width of USA highway lanes which is 3.7 meters

    return leftLane, rightLane, vehiclePosition

def warpToOriginal(lines, warpMatrix):
    inverseTransform = np.linalg.inv(warpMatrix)  # Inverse matrix

    # Transform each point of the lines using the inverse perspective matrix
    originalLines = []
    for line in lines:
        x1, y1, x2, y2 = line
        points = np.array([[x1, y1], [x2, y2]], dtype='float32').reshape(-1, 1, 2)
        transformedPoints = cv2.perspectiveTransform(points, inverseTransform)
        transformedPoints = transformedPoints.reshape(-1, 2)
        originalLines.append((int(transformedPoints[0][0]), int(transformedPoints[0][1]),
                               int(transformedPoints[1][0]), int(transformedPoints[1][1])))
    return originalLines

def drawLinesOnImage(originalImage, lines):
    # Create a copy of the original image to draw on
    outputImage = originalImage.copy()

    # Loop through each line and draw it on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(outputImage, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)  # Red lines

    return outputImage

if __name__ == '__main__':
    main()