import cv2
import numpy as np
import os
from utils.changeColorSpace import changeBGRtoHSV, changeBGRtoYCBCR, changeBGRtoCIELAB, changeBGRtoCIELUV
from multiResDescriptors import generateMultiResDescriptors
from get2biggerAreas_contours import getBiggestContours



def computeDescriptors(imagesPath, outputPath, colorSpace, numBins, histGenFunc, levels, textBoxes = None,
                       backgroundMaskDir = None, multipleImages = "no"):
    """ This function computes the descriptors of the images from the input path 
    and save them in the output path.
    

    Parameters
    ----------
    imagesPath : string
        The path were input images are.
    outputPath : string
        The path were descriptors will be saved.
    colorSpace : string
        The color space were the descriptors will be generated. 
        rgb, hsv, cielab, cieluv, ycbcr are the options.
    numBins: int
        The number of bins in each dimension
    histGenFunc: function
        The function to generate the descriptors
    levels: int
        The number of multidimensional levels
    backgroundMaskDir : str
        Path where binary mask to know which pixel is related to the background are.
        Default: None  -> No background
    textBox: numpy array (np.int)
        List of the text box coordinates

    Returns
    -------


    """
    
    
    # Get file names of the database
    files = os.listdir(imagesPath)
    
    for file in files:
        # Check if it is an image
        if file[-4:] == ".jpg":
            imageNum = int(file[:-4].split("_")[-1])
            image = cv2.imread(imagesPath + file)
            
            if backgroundMaskDir is None:
                # Take into account every pixel
                mask = np.zeros(image.shape[:2], dtype = np.uint8) + 255
            else:
                mask = cv2.imread(backgroundMaskDir + file[:-4] + ".png", cv2.IMREAD_GRAYSCALE)

            # Convert the image into the new color space
            if colorSpace == "hsv":
                image = changeBGRtoHSV(image)
            elif colorSpace == "cielab":
                image = changeBGRtoCIELAB(image)
            elif colorSpace == "cieluv":
                image = changeBGRtoCIELUV(image)
            elif colorSpace == "ycbcr":
                image = changeBGRtoYCBCR(image)
            
            if multipleImages != "no":
                
                boxes = getBiggestContours(mask)
                # Empty mask
                mask = np.zeros(image.shape[:2], dtype = np.uint8) + 255
                    
            else:
                boxes = [(0,0,image.shape[1]-1, image.shape[0]-1)]
                
            for i, box in enumerate(boxes):
                xMinP, yMinP, xMaxP, yMaxP = box
                
                paintingNew =  image[yMinP: yMaxP + 1, xMinP: xMaxP + 1]
                maskNew = mask[yMinP:yMaxP + 1, xMinP:xMaxP + 1]
                
                if not(textBoxes is None):
                    # Add text box pixels to the mask
                    textBox = textBoxes[imageNum][i]
                    
                    xMin = textBox[0][0] - xMinP
                    yMin = textBox[0][1] - yMinP
                    
                    xMax = textBox[2][0] - xMinP
                    yMax = textBox[2][1] - yMinP
                    
                    maskNew[yMin:yMax + 1, xMin:xMax + 1] = 0
                    
                    
                descriptor = generateMultiResDescriptors(paintingNew, maskNew, levels, histGenFunc, numBins)
                
                descriptorPath = outputPath + file[:-4] + "_" + str(i) + ".npy"
                np.save(descriptorPath, descriptor)