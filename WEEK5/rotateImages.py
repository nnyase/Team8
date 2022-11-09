import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from utils.getBiggestAreasContours import getBiggestContours
from utils.transformPoints import rotatePoints
from utils.managePKLfiles import store_in_pkl
from copy import deepcopy

def getFrames(maskImages, angles, output):
    """
    This function computes the frame points of the paintings

    Parameters
    ----------
    maskImages : str
        Path were rotated images background mask are located.
    angles : list
        Original rotation angle of each image.
    output : str
        Path to store the frames.pkl file.

    Returns
    -------
    None.

    """
    angleIndex = 0
    
    frames = []
    
    for imageP in sorted(os.listdir(maskImages)):
        
        if imageP[-4:] == ".png":
            
            # Read mask
            img = cv2.imread(maskImages + imageP, cv2.IMREAD_GRAYSCALE)
            
            # Get rotation angle
            angle = angles[angleIndex]
            rotationAngle = angle
            
            if angle > 90:
                angle = angle - 180
            
            angleIndex += 1
            
            # Find contours
            imageFrames = []
            boxes = getBiggestContours(img)
            
            imageCenter = ((img.shape[1]-1)/2, (img.shape[0]-1)/2)
            
            for box in boxes:
                xMin, yMin, xMax, yMax = box
                
                points =[[xMax, yMax] , [xMax,yMin], [xMin, yMin], [xMin, yMax]]
                
                
                # notPoint = deepcopy(points)
                
                # Rotate points to get the original coordinates
                transformedPoints = rotatePoints(angle, points, imageCenter)
                
                # imgOr = cv2.imread("../../WEEK5/qsd1_w5/" + imageP[:-4] + ".jpg")
                # for i, point in enumerate(transformedPoints):
                #     cv2.circle(imgOr, (int(point[0]), int(point[1])), 0, [255,0,0], 50)
                #     cv2.circle(img, (int(notPoint[i][0]), int(notPoint[i][1])), 0, [255,0,0], 50)
                # plt.imshow(imgOr)
                # plt.show()
                # plt.imshow(img)
                # plt.show()
                
                imageFrames.append([rotationAngle, transformedPoints])
            
            frames.append(imageFrames)
    
    # Store pkl
    store_in_pkl(output, frames)

def rotateImages(inputPath, outputPath, angles):
    """
    This function rotates and saves the images of the input path.

    Parameters
    ----------
    inputPath : str
        Original images path.
    outputPath : str
        New rotated images path.
    angles : list
        Rotation angle of each image.

    Returns
    -------
    None.

    """
    angleIndex = 0
    
    for imageP in sorted(os.listdir(inputPath)):
        
        if imageP[-4:] == ".jpg":
            
            img = cv2.imread(inputPath + imageP)
            
            angle = angles[angleIndex]
            if angle > 90:
                angle = angle - 180
            angle = -angle
            
            angleIndex += 1
            
            # Create rotation matrix
            imageSize = (img.shape[1] , img.shape[0])
            imageCenter = ((img.shape[1]-1)/2, (img.shape[0]-1)/2)
            M = cv2.getRotationMatrix2D(imageCenter, angle, 1.0)
            rotatedImage = cv2.warpAffine(img, M, imageSize, borderMode = cv2.BORDER_REPLICATE)
            
            cv2.imwrite(outputPath + imageP, rotatedImage)

def getImagesRotation(imagesPath):
    """
    This function finds the rotation angle of each image in the input path

    Parameters
    ----------
    imagesPath : str
        Background mask images path.

    Returns
    -------
    angles : list
        Rotation angles of the images.

    """
    angles = []
    for imageP in sorted(os.listdir(imagesPath)):
        
        if imageP[-4:] == ".png":
            
            img = cv2.imread(imagesPath + imageP, cv2.IMREAD_GRAYSCALE)
            angle = getImageRotation(img)
            angles.append(angle)
    
    return angles


def getImageRotation(img):
    """
    This function finds the rotation angle of the input images.

    Parameters
    ----------
    img : numpy array (np.uint8)
        Input image.

    Returns
    -------
    angle : float
        Rotation angle in degrees.

    """
    
    # Convert to grayscale
    #imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find edges (Canny)
    imageEdges = cv2.Canny(img, 80, 100)
    
    
    # Hough transform to find lines
    lines = cv2.HoughLinesP(imageEdges, 1, np.pi / 180.0, 100, minLineLength=100, maxLineGap=50)
    
    # Lines
    correctLinesAngles = []
    for i in range(lines.shape[0]):
        
        x0,y0,x1,y1 = lines[i,0,:]
        
        # Get line angle in degrees
        angle = np.arctan2(y1-y0,x1-x0)*(180/np.pi)
        
        angle = -angle
        # Positive angles
        if angle < 0:
            angle += 180
        
        # No vertical lines
        if not(angle > 60 and angle < 120):
        
            correctLinesAngles.append(angle)
    
    # Get angle
    if len(correctLinesAngles) == 0:
        angle = 0.0
    else:
        if len(correctLinesAngles) % 2 == 0:
            correctLinesAngles.append(correctLinesAngles[0])
            
        angle = np.median(correctLinesAngles)
    
    
    return angle
    
    
    