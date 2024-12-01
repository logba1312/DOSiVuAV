import numpy as np
import cv2

def fitPolynomials(filteredLines, imageHeight, leftLineX, rightLineX, threshold=50):
    # Separate lines into left and right groups
    leftLines = []
    rightLines = []

    for line in filteredLines:
        x1, y1, x2, y2 = line
        xCoord = (x1 + x2) // 2  # Average x-coordinate
        if abs(xCoord - leftLineX) <= threshold:  # Group lines near the left lane
            leftLines.append(line)
        elif abs(xCoord - rightLineX) <= threshold:  # Group lines near the right lane
            rightLines.append(line)

    # Fit polynomials for both left and right lines
    leftXVals, leftYVals, leftCoeff = fitSinglePolynomial(leftLines, imageHeight)
    rightXVals, rightYVals, rightCoeff = fitSinglePolynomial(rightLines, imageHeight)

    return (leftXVals, leftYVals, leftCoeff), (rightXVals, rightYVals, rightCoeff)

# Helper function to fit a polynomial for a given set of lines
def fitSinglePolynomial(lines, imageHeight):
    xPoints = []
    yPoints = []
    for line in lines:
        x1, y1, x2, y2 = line
        xPoints.extend([x1, x2])
        yPoints.extend([y1, y2])
    
    if len(xPoints) > 2:
        coefficients = np.polyfit(yPoints, xPoints, deg=2)
        yVals = np.linspace(0, imageHeight - 1, num=imageHeight)
        xVals = np.polyval(coefficients, yVals)
        return xVals, yVals, coefficients
    else:
        return None, None, None
    
def calculateCurvature(polyCoeffs, yEval, ymPerPix=30/720, xmPerPix=3.7/235):
    # Extract coefficients
    A = polyCoeffs[0]  # Quadratic coefficient
    B = polyCoeffs[1]  # Linear coefficient

    # Scale the polynomial to real-world values
    A_real = A * (xmPerPix / (ymPerPix**2))
    B_real = B * (xmPerPix / ymPerPix)

    # Calculate the curvature radius
    curvatureRadius = ((1 + (2 * A_real * yEval * ymPerPix + B_real) ** 2) ** 1.5) / abs(2 * A_real)
    return curvatureRadius