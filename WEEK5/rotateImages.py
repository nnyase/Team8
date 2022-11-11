import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import math
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

def getPoints(img):
    """
This function finds the points of the bottom Line of the painting.

Parameters
----------
img : cv2.imread (grayScaled image)

Returns
-------
pointCC,pointDD :
    Line points.

"""
  
    # Detecting contours in image.
    contours, _= cv2.findContours(img, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
  
# draws boundary of contours.
    cv2.drawContours(img, [approx], 0, (255, 255, 255), 5) 
   
# Used to flatted the array containing
 # the co-ordinates of the vertices.
    n = approx.ravel() 
    i = 0
    pointA=[]
    pointB=[]
    pointC=[]
    pointD=[]
    vector=[]
    auxVector=[]
    for j in n :
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]
  
            #  co-ordinates four points of the rectangle with no sequence    
            if(i == 0):
                pointA=(x,y)
                auxVector.append(pointA)
                vector.append(pow(((x*x)+(y*y)),(1/2)))
            if(i == 2):
                pointB=(x,y)
                auxVector.append(pointB)
                vector.append(pow(((x*x)+(y*y)),(1/2)))
            if(i == 4):
                pointC=(x,y)
                auxVector.append(pointC)
                vector.append(pow(((x*x)+(y*y)),(1/2)))
                
            if(i == 6):
                pointD=(x,y)
                auxVector.append(pointD)
                vector.append(pow(((x*x)+(y*y)),(1/2)))

        i = i + 1        
    tmp = max(vector)
    index = vector.index(tmp)
    tmp1 = min(vector)
    index2 = vector.index(tmp1)
    
    #detect nearest and farthest points
    # and use the farthest point as  second point of our line
    if index==0:
        pointDD=pointA
    if index==1:
        pointDD=pointB
    if index==2:
        pointDD=pointC
    if index==3:
        pointDD=pointD
        
    if index2==0:
        pointAA=pointA
    if index2==1:
        pointAA=pointB
    if index2==2:
        pointAA=pointC
    if index2==3:
        pointAA=pointD 
    
    auxIndex=[]
    
    
    for j in range(len(auxVector)):
        if auxVector[j] != pointAA and auxVector[j] != pointDD:
            auxIndex.append(auxVector[j])
    # evaluate the last 2 points to find the first point of the line
    if auxIndex[0][0] < auxIndex[1][0]:
        pointCC=auxIndex[0]
    else:
        pointCC=auxIndex[1]

        
    return pointCC,pointDD
  

def getImageRotation_2(img):
    """
This function finds the angle of rotation of a signle paiting 

Parameters
----------
img : cv2.imread (grayScaled image)

Returns
-------
Angle : angle of rotation 
    L

"""
    
    pointA,pointB=getPoints(img)
    if pointA[1]==pointB[1]:
        angle= 0
    else:
        x1=pointA[0]
        y1=pointA[1]
        x2=pointB[0]
        y2=pointB[1]
        a= x2-x1
        if y1>y2:    
            b= y1-y2
            c= pow(((a*a)+(b*b)),(1/2))
            c= pow(((a*a)+(b*b)),(1/2))
            angle=math.degrees(math.acos(a/c))
        else:
            b= y2-y1
            c= pow(((a*a)+(b*b)),(1/2))
            angle= 180-math.degrees(math.acos(a/c))
        
    
    
    return angle

def getImagesRotation_2(imagesPath):
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
            angle = getImageRotation_2(img)
            angles.append(angle)
    
    return angles


    
    
    
