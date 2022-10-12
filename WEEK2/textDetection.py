import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def detectText(image, multiplePaintings = False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageS = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]<20
    

    kernel = np.ones((3,3))
    dilateImage = cv2.dilate(imageGray, kernel, 1)
    erodeImage = cv2.erode(imageGray, kernel, 1)

    morphologicalGradient = dilateImage - erodeImage
    
    morphologicalGradient = morphologicalGradient * imageS 
    bigGradient = ((morphologicalGradient>150)*255).astype(np.uint8)
    
    
    kernel = np.ones((7,7))
    bigGCD = cv2.dilate(bigGradient, kernel, 1)
    kernel = np.ones((9,9))
    bigGCE = cv2.erode(bigGCD, kernel, 1)
    #kernel = np.ones((5,5))
    #bigGCO = cv2.morphologyEx(bigGCC, cv2.MORPH_OPEN, kernel)
    
    return detectRectangle(bigGCE)

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

def detectTextBoxes(inputPath, outputPath):
    for file in os.listdir(inputPath):
        if file[-4:] == ".jpg":
            
            image = cv2.imread(inputPath + file)
            detection = detectText(image)
            cv2.imwrite(outputPath + file, detection)
            
    
            

inputPath = "../../WEEK2/qsd1_w2/"
outputPath = "./textMasks/"

detectTextBoxes(inputPath, outputPath)


        