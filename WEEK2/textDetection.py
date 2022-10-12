import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from utils.managePKLfiles import store_in_pkl

def detectText(image):
    """ This functions returns the BBox of the text box in the input image given.
    

    Parameters
    ----------
    image : numpy array (uint8)
        image of one painting (without background) in BGR color space.

    Returns
    -------
    list
        box coordinates.

    """
    # Get grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get saturation (S)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageS = imageHSV[:,:,1]
    
    # Threshold saturation with small values
    imageSmallS = imageS < 20
    
    
    # Morphological gradients
    kernel = np.ones([3,3])
    dilateImage = cv2.dilate(imageGray, kernel, 1)
    erodeImage = cv2.erode(imageGray, kernel, 1)
    morphologicalGradient = dilateImage - erodeImage
    
    # Gradients of only small saturation
    morphologicalGradient = morphologicalGradient * imageSmallS 
    
    # Binarize
    _, binary = cv2.threshold(morphologicalGradient, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Set edges to 0
    binary[0,:] = 0
    binary[binary.shape[0]-1,:] = 0
    binary[:,0] = 0
    binary[:, binary.shape[1]-1] = 0
    
    # Put the letters together
    kernel = np.ones([1,int(image.shape[1]/20)])
    binaryClose1 = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, kernel)
    
    # Remove noise
    kernel = np.ones([5,5])
    binaryOpen = cv2.morphologyEx(binaryClose1,cv2.MORPH_OPEN, kernel)
    
    
    # Put the words together
    kernel = np.ones([1,int(image.shape[1]/5)])
    binaryClose = cv2.morphologyEx(binaryOpen,cv2.MORPH_CLOSE, kernel)
    
    # Remove noise
    kernel = np.ones([5,5])
    binaryOpen = cv2.morphologyEx(binaryClose,cv2.MORPH_OPEN, kernel)
    
    # Set edges to 0
    binaryOpen[0,:] = 0
    binaryOpen[binaryOpen.shape[0]-1,:] = 0
    binaryOpen[:,0] = 0
    binaryOpen[:, binaryOpen.shape[1]-1] = 0
    
    
    # Get biggest contour
    contour = getBiggestContour(binaryOpen)
    
    # Compute BBox
    if not (contour is None):
        # Get BBox from contour
        minX = int(np.min(contour[:,0,0]))
        maxX = int(np.max(contour[:,0,0]))
        
        diffX = int((maxX - minX)*0.03)
        minX = max(0, minX-diffX)
        maxX = min(image.shape[1], maxX + diffX)
        
        
        minY = int(np.min(contour[:,0,1]))
        maxY = int(np.max(contour[:,0,1]))
        
        diffY = int((maxY - minY)*0.15)
        minY = max(0, minY-diffY)
        maxY = min(image.shape[0], maxY + diffY)
    
    
    else:
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0

    
    return [np.array([minX, minY]), np.array([minX, maxY]), np.array([maxX, maxY]), np.array([maxX, minY])]


def getBiggestContour(binaryImage):
    """
    This function returns the bigges contour in the binary input image.

    Parameters
    ----------
    binaryImage : numpy array (uint8)
        Binary image to get contours from.

    Returns
    -------
    biggestContour : numpy array (uint8)
        Found biggest contour.

    """
    biggestContour = None
    biggestArea = -1    
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > biggestArea:
            
            biggestArea = area
            biggestContour = contour
            
            
    
    return biggestContour

"""
def detectText1(image, multiplePaintings = False):
    
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageS = imageHSV[:,:,1]
    
    imageSmallS = imageS < 20
    
    kernel = np.ones([3,3])
    dilateImage = cv2.dilate(imageGray, kernel, 1)
    erodeImage = cv2.erode(imageGray, kernel, 1)

    morphologicalGradient = dilateImage - erodeImage
    
    morphologicalGradient = morphologicalGradient * imageSmallS 
    
    _, binary = cv2.threshold(morphologicalGradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones([1,int(image.shape[1]/10)])
    binaryClose = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, kernel)
    
    kernel = np.ones([5,5])
    binaryOpen = cv2.morphologyEx(binaryClose,cv2.MORPH_OPEN, kernel)
    #bigGradient = ((morphologicalGradient>130)*255).astype(np.uint8)
    #morphologicalGradient = bigGradient * imageSmallS
    
    #kernel = np.ones((7,7))
    #bigGCD = cv2.dilate(bigGradient, kernel, 1)
    #kernel = np.ones((9,9))
    #bigGCE = cv2.erode(bigGCD, kernel, 1)
    #kernel = np.ones((5,5))
    
    #bigGCO = cv2.morphologyEx(bigGCC, cv2.MORPH_OPEN, kernel)
    
    return binaryOpen#detectRectangle(bigGCE)


def detectRectangle(textImage):
    iMin = textImage.shape[0]-1
    iMax = 0
    jMin = textImage.shape[1]-1
    jMax = 0
    
    for i in range(textImage.shape[0]):
        for j in range(textImage.shape[1]):
            
            if textImage[i,j] == 255:
                if i > iMax:
                    iMax = i
                    
                if i < iMin:
                    iMin = i
                    
                if j > jMax:
                    jMax = j
                
                if j < jMin:
                    jMin = j
    
    textImage[iMin:iMax+1, jMin:jMax+1] = 255
    
    return textImage


def distance(color1, color2):
    
    distance = color1.astype(int) - color2.astype(int)
    
    value = np.sqrt(distance[0]**2 + distance[1]**2 + distance[2]**2)
    
    return value
    

def postProcessMask(image, i, j):
    
    originalColor = image[i,j]
    
    checked = np.zeros(image.shape[:2], dtype = np.uint8)
    background = np.zeros(image.shape[:2], dtype = np.uint8) 
    
    queue = [(i,j)]
    checked[i,j] = 255
    background[i,j] = 255
    
    distanceMax = 10
    
    while len(queue) > 0:
        i, j = queue.pop()
        
        if i + 1 < image.shape[0]:
                
                if checked[i+1,j] == 0:
                    checked[i+1,j] = 255
                    if distance(image[i+1,j], originalColor) < distanceMax:
                        background[i + 1, j] = 255
                    
                        queue.append((i+1,j))
                    
        if i - 1 > 0:
            if checked[i-1,j] == 0:
                checked[i-1,j] = 255
                if distance(image[i-1,j], originalColor) < distanceMax:
                    background[i - 1, j] = 255
                
                    queue.append((i-1,j))
        
        if j + 1 < image.shape[1]:
                
                if checked[i,j+1] == 0:
                    checked[i,j+1] = 255
                    if distance(image[i,j+1], originalColor) < distanceMax:
                        background[i, j+1] = 255
                    
                        queue.append((i,j+1))
                    
        if j - 1 > 0:
            if checked[i,j-1] == 0:
                checked[i,j-1] = 255
                if distance(image[i,j-1], originalColor) < distanceMax:
                    background[i, j-1] = 255
                
                    queue.append((i,j-1))
                    

    return background
"""
def detectTextBoxes(inputPath, outputPath):
    results = []
    for file in os.listdir(inputPath):
        if file[-4:] == ".jpg":
            
            image = cv2.imread(inputPath + file)
            detection = detectText(image)
            results.append([detection])
    
    store_in_pkl(outputPath + "text_boxes.pkl", results)
            
    
            

inputPath = "../../WEEK2/qsd1_w2/"
outputPath = "./textBoxes/"

detectTextBoxes(inputPath, outputPath)
