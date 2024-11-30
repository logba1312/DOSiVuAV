# DOSiVuAV
## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output/chessBoard.jpg "Drawn Chessboard Corners"
[image2]: ./output/Distortion-corrected.jpg "Distortion-corrected"
[image3]: ./output/warpedImage.jpg "Warped Image"
[image4]: ./output/SobleBinary.jpg "Sobel Binary"
[image5]: ./output/colorMask.jpg "Color mask"
[image6]: ./output/combined.jpg "Sobel + Color Mask"
[image7]: ./output/morphologicalClosing.jpg "Morphological Closing"
[image8]: ./output/histogram.jpg "Histogram With Peaks"
[image9]: ./output/HoughLines.jpg "Hough lines with vehicle offset"
[image10]: ./output/polyImage.jpg "Polyfit lines with vehicle offset"
[video1]: ./output/finalVideo.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

First things first, in order to calibrate the camera we need to set up a few parameters. To be able to detect the chess board that we are using for calibration, we need to know it's dimensions(widht and height). By looking at one of the example images, we conclude that the chess board is of dimensions 7x10. However after attempting to find the corners of the chessboard and drawing the result on the original picture, I noticed that the algorithm didn't work. After looking into it, I figured out that the width and heght are supposed to be for the corners and not the squares themselves. After running the function again (cv2.findChessboardCorners()) with dimensions set at 6x9, the corners were succesfully found and drawn on the image, as shown on the picture below. To make the findChessboardCorners function more precise we turn the original image into grayscale, removing the colors from the image. This makes the function much faster as it now doesn't need to do edge detection in all three (RGB) channels but only in one (only checking the intensity of the pixels). This is a common preprocessing step and will be used multiple times in this project. 
The findChessboardCorners function returns the coordinates of the corners of the chess board found on the image but to be able to undistort the camera we also need refference poinst that show where these corners should actually be. For that we create arrays of poinst that are the same for each image because the board is the same in each of them. The objectPoints variable is this array for the image. After finding the corners on the image we refine these positions with cv2.cornerSubPix which is a function used to refine the locations of corners (or interest points) in an image to sub-pixel accuracy. After adding all these arrays of points of each image to another array we can porceed with the calculation. We do the calculation using cv2.calibrateCamera and use the objectPonts array and the corner points array (imagePointsArray in the code) as well as the image size as its paramaters. The function then returns intrinsic parameters and the distortion coefficients, as well as some other values that were not used further in the project(reprojection error, rotation vectors, translation vectors). These values are saved in a file called "calib.npz" and are later used by the undistort function.
The whole purpose of the calibration is to remove the so called "fish-eye lense" effect of the camera, so that we could further work with an image that is more true to how the real world is. The undistorted image can be seen in the "Distortion-corrected" image in the next sub-section below.

![image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The first step for undistorting an image is to read the output of the camera calibration. Using the cv2.getOptimalNewCameraMatrix function we calculate the new camera matrix for the input image and then using cv2.undistort we undistort the image using the input image, matrix, distortion and new camera matrix as its parameters.

![image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code described below an be found in ./src/warpImage.py
After undistorting the image, the next step is to warp the image so that it would appear that the image was taken from a birds eye perspective. After a lengthy period of trial and error, I made a function called selectPoints that showed me what the pixel coordiantes were when I clicked on the image. Using an image where there are straight lines, I selected pixels that were on the line because the output image then would have to have vertical lines. 
It should be mentioned that not all input images are of the same resolution, which messes up the warping of the image. That's why before warping the image I upscaled the smaller pictures to 1280x720. This is the resolution of the camera that was used for the calibration. For upscaling I used cv2.resize with interpolation=cv2.INTER_LINEAR because this interpolation method is relatively fast and well suited for upscaling images.
After selecting the source points, the destination points also needed to be selected, and once again using trial and error I ended up with the coordinates shown in the code below.

```python
    sourcePoints = np.float32([
        [600, 455],      # Top-left corner
        [718, 455],      # Top-right corner
        [350, 621],      # Bottom-left corner
        [984, 621]      # Bottom-right corner
    ])
    destinationPoints = np.float32([
        [350 + 200, 0],         # Top-left corner
        [984 - 200, 0],         # Top-right corner
        [350 + 200, 720],         # Bottom-left corner
        [984 - 200, 720]          # Bottom-right corner
    ])
```

With these points I was able to calculate the perspective transformation matrix using cv2.getPerspectiveTransform. Finally, using this matrix, the image could be warped with the cv2.warpPerspective function.
With hindsight, the destination and source points should have been chosen differently, as the camera is positioned slightly to the right of the center of the car. The not so ideal choice of poinst results in a bad reverse-warp of the detected lanes, however that is only for show. Functionality wise it still does the job well.

![image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code described below an be found in ./src/binaryImage.py
I wanted to create a binary image that made the lanes look as thick as possible and in order to achieve that, I used multiple algorithms combined. The algorithms in question are Sobel edge detection, threshold, binary masking, and morpological closing. I chose the Sobel function for edge detection because it usually is faster than Canny, and speed is important in embeded systems since the usually don't have powerful hardware.  
To prepare for the Sobel algorithm, first the warped image was turned into a gray image using cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for the same reason as explained earlier. Secondly a Gaussian blur was applied in order to remove any existing noise that would impact the edge detection. Finally I used the cv2.Sobel function to calculate the gradient but only on the x-axis, because this would ignore all horisontal lines and detect vertial ones, which is exactly what we need. The output from the Sobel function is still not a binary image, so a binary threshold is applied to the image (but not before scaling the image to uint8 - values from 0 to 255). By testing multiple threshold values, 50 was showing great results. 
![image4]
On some images the Sobel fumction detected the lanes very well but on others it struggled to detect the lanes. Therefore an aditional detection is needed. The method in question is applying a color mask on the image. We know that the color of the lanes are either white or yellow, so we search for all white and yellow pixels in the image. First we transform the picture from RGB to HLS. By separating color (Hue) from brightness (Lightness), we can focus on color detection regardless of variations in lighting conditions. For example, a yellow line on a bright road and the same yellow line in shadow will have different RGB values but similar Hue values. After that we use the cv2.inRange function which selects all the pixels that are within a given range. The ranges used in this project are the following:
```python
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        lowerYellow = np.array([18, 130, 100])
        upperYellow = np.array([30, 255, 255])
        yellowMask = cv2.inRange(hls, lowerYellow, upperYellow)

        lowerWhite = np.array([0, 200, 0])
        upperWhite = np.array([255, 255, 255]) 
        whiteMask = cv2.inRange(hls, lowerWhite, upperWhite)
```
On the image below we can see that all the lane lines were detected succesffuly. Even the lines from the lane next to the one we are in. This could prove useful when merging from one lane to another, but for this project we are interested only in the lines of the lane that the car is located in.  
![image5]
To ensure that all the vertical lines in the image were detected, we combine the binary Sobel image and the binary image from the color mask. 
![image6]
The Sobel edge detection detects two edges on a single lane line which is not ideal, so to fix that I fill the gaps using cv2.morphologyEx.
![image7]
On the resulting image we can see that there is still a lot of unnecesary lines that were detected that aren't lane lines. This issue is solved by doing a hystogram in the next step. 
I would like to note that there are more ways to ignore the unnecessary lines like applying a ROI or using different line detection methods like Canny or using different values for thresholds, but I wanted to keep as much information as possible from the image.  

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code described below an be found in ./src/histogramWithPeaks.py
This part of the code is where we sperate the useful data from the rest. Going pixel by pixel on the x-axis we sum up the number of white pixels in the binary image. Wit this we get a histogram showing the number of detected pixels per column. For vertical lanes the number of detected pixels will be higher than for any other lines. In some examples we have cars in other lanes that are detected as diagonal lines and since we calculate the sum of the column, the histogram will show a small value where the car is. The largest value will be where there are full straight lines and smaller values where the lines are cut. The exact pixel where the lines are are calculated with scipy.signal.find_peaks, where we can set the parameter of what is considered a peak. I set the value of anything over 100 that is at least 50 pixels away from each other on the x-axis.
On the image below we can see that the yellow line has the largest value, followed by the edge between the wall and the road (with further development this can be very usefull to avoid the edge of the road which is not marked with a line), and then the rest of the lanes. For lane centering capabilities we only need the lane lines of our lane, which are allways the closest from the lef and right. Using this type of selecting lanes makes the program capable of centering in a new lane after merging.

![image8]

In ./src/main.py we take all the detected peaks and only coose the ones that are left and right of the center that I approximated is around pixel 665 of the warped image. To calculate the center and do more precise warping of the image, the dimensions of the car and the offset of the camera from the center of the car would be usefull. The position of the vehicle is calculated by subracting the center pixel(665) from the middle point of the left and right lines. If the number is negatve the care is more to the left and vice versa. To convert the pixels to meters we multiply the vehicle offset in pixels with 0.006 which is calculated by dividing the number of pixels between the twto lanes, with the average lane width of USA highway lanes which is 3.7 meters.

To actually identify the lane line pixels I used the Hough Line Transform, and set its parameters to detect vertical lines. This is done by setting rho to 1 and theta to pi/180. To make sure that only the necessary lines are detected, i set the input image to be the binary image that we created earlier. After the algorithm calculates all the lines from the binary image I only select the lines detected Around the left and right line coordinates. After all these are the lines that we are interested in. 

![image9]

Now that we have the lines from both lanes, we can calculate a polynomial to fit in each lane line. This is done in the ./src/fitPolinomial.py module. First the left and right lane are seperated into two arrays. After that, for each lane line, we put all the points from the Hough lines through the np.polyfit function. This function performs a polynomial curve fit through the points of th edetected lines. We Ensure that the line will be curved by putting the deg parameter to 2 (2nd degree polynomial y=ax^2+bx+c).

![image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!
![video1]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

