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
    fileName = 'test1.jpg'
    # fileName = 'challenge01.mp4'

    displayImageResult(fileName)
    # displayVideoResult(fileName)

    

def displayImageResult(imgName):
    # Open image
    img = cv2.imread(findFile(imgName, Path(__file__).parent.parent))
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
    laneCooridnates = hist.histogramWithPeaks(binaryImage)
    leftLane = 0
    rightLane = 0

    for point in laneCooridnates:
        if point <= 665:
            leftLane = point
        else:
            rightLane = point
            break

    vehiclePosition =  ((leftLane + rightLane) / 2 - 665) * 0.006 # vehicle offset from center in meters

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(warpedImg)
    plt.scatter([leftLane, rightLane], [200, 200], color='red', label=['Left', 'Right'])
    plt.scatter([leftLane, rightLane], [700, 700], color='red', label=['Left', 'Right'])
    plt.plot([leftLane, leftLane], [200, 720], color='red', linewidth=2)
    plt.plot([rightLane, rightLane], [200, 720], color='red', linewidth=2)
    plt.title('Location of left and right lane')
    plt.text(5, 35, f"Vehicle offset from center: {vehiclePosition:.3f}m", fontsize=12, color='yellow')

    filteredLanes = dl.detectVerticalLines(binaryImage, leftLane, rightLane)

    # Visualize the result
    for x1, y1, x2, y2 in filteredLanes:
        cv2.line(warpedImg, (x1, y1), (x2, y2), (0, 255, 0), 5)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(warpedImg, cv2.COLOR_BGR2RGB))
    plt.title("Detected Vertical Lines")
    plt.text(5, 35, f"Vehicle offset from center: {vehiclePosition:.3f}m", fontsize=12, color='white')
    plt.tight_layout()
    plt.show()

def displayVideoResult(videoName):
    # Path to the video file
    video_capture = cv2.VideoCapture(findFile(videoName, Path(__file__).parent.parent))

    # Check if the video file was opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        exit()

    out = cv2.VideoWriter("finalVideo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280 * 3, 720))

    # Process video frames
    while True:
        ret, frame = video_capture.read()  # Read a frame from the video

        # Break the loop if no frame is returned (end of video)
        if not ret:
            break

        # Change image size if the resolution is different from camera resolution
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            newWidth, newHeight = 1280, 720
            frame = cv2.resize(frame, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)

        # Process the frame
        undistortedImage = undistort.undistort(frame)  # Use your undistort function
        warpedImg, warpMatrix = wi.warpImage(undistortedImage)    # Use your warp function
        binaryImage = bi.binaryImage(warpedImg)  # Binary and Sobel images
        laneCoordinates = hist.histogramWithPeaks(binaryImage)  # Histogram peak detection
        leftLane, rightLane = 0, 0

        for point in laneCoordinates:
            if point <= 650:
                leftLane = point
            else:
                rightLane = point
                break

        vehiclePosition = ((leftLane + rightLane) / 2 - 665) * 0.006  # Vehicle offset from center in meters

        # Detect vertical lines
        filteredLanes = dl.detectVerticalLines(binaryImage, leftLane, rightLane)

        # Draw detected lanes on the warped image
        for x1, y1, x2, y2 in filteredLanes:
            cv2.line(warpedImg, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Display the result using OpenCV (you can save the frames if needed)
        displayImg = cv2.cvtColor(warpedImg, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        cv2.putText(
            displayImg,
            f"Vehicle offset: {vehiclePosition:.3f} m",  # Text content
            (5, 500),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale
            (255, 255, 0),  # Font color (Yellowish)
            2,  # Thickness
            cv2.LINE_AA,  # Line type
        )
        # cv2.imshow("Processed Frame", displayImg)
        # cv2.imshow("Original Image", frame)        

        originalLines = warpToOriginal(filteredLanes, warpMatrix)
        resultImage = drawLinesOnImage(frame, originalLines)
        cv2.imshow("Lines on Original Image", np.hstack((frame, displayImg, resultImage)))
        out.write(np.hstack((frame, displayImg, resultImage)))

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

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