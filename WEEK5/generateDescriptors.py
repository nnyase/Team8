import cv2
import numpy as np
import os
from utils.changeColorSpace import changeBGRtoHSV, changeBGRtoYCBCR, changeBGRtoCIELAB, changeBGRtoCIELUV
from multiResDescriptors import generateMultiResDescriptors
from utils.getBiggestAreasContours import getBiggestContours
from utils.hog import createHoGdescriptor
from utils.dctDescriptors import createDCTdescritor
from utils.wavelet import createWaveletDescriptor
from utils.lbpHist import createLBPhistogram
from localDescriptor import generateLocalDescriptors


def computeColorDescriptors(imagesPath, outputPath, colorSpace, numBins, histGenFunc, levels, textBoxes = None,
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
                    
                    xMin = textBox[0] - xMinP
                    yMin = textBox[1] - yMinP
                    
                    xMax = textBox[2] - xMinP
                    yMax = textBox[3] - yMinP
                    
                    maskNew[yMin:yMax + 1, xMin:xMax + 1] = 0
                    
                    
                descriptor = generateMultiResDescriptors(paintingNew, maskNew, levels, histGenFunc, numBins)
                
                descriptorPath = outputPath + file[:-4] + "_" + str(i) + ".npy"
                np.save(descriptorPath, descriptor)


def computeTextureDescriptors(imagesPath, outputPath, textureType, levels, numFeatures, textBoxes = None,
                       backgroundMaskDir = None, multipleImages = "no"):
    """
    This function computes the texture descriptors of the images in the given folder and store them in the output path.

    Parameters
    ----------
    imagesPath : str
        Path of the images.
    outputPath : str
        Path where text descriptors will be saved.
    textureType : str
        Which texture type to use.
    levels : int
        Levels for multires descriptors.
    numFeatures : int
        Number of features to get for each block.
    textBoxes : list, optional
        Text boxes detected. The default is None.
    backgroundMaskDir : str, optional
        Path where backkground mask is stored. The default is None.
    multipleImages : str, optional
        Yes if the image can contain more than one painting. The default is "no".

    Returns
    -------
    None.

    """
    
    # Get texture function
    if textureType == "lbp":
        fDesTexture = createLBPhistogram
    elif textureType == "dct":
        fDesTexture = createDCTdescritor
    elif textureType == "hog":
        fDesTexture = createHoGdescriptor
    elif textureType == "wavelet":
        fDesTexture = createWaveletDescriptor
        
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
                    
                    xMin = textBox[0] - xMinP
                    yMin = textBox[1] - yMinP
                    
                    xMax = textBox[2] - xMinP
                    yMax = textBox[3] - yMinP
                    
                    maskNew[yMin:yMax + 1, xMin:xMax + 1] = 0
                 
                    paintingNew[yMin:yMax + 1, xMin:xMax + 1] = [0, 0, 0]
                    
                descriptor = generateMultiResDescriptors(paintingNew, maskNew, levels, fDesTexture, numFeatures)
                
                descriptorPath = outputPath + file[:-4] + "_" + str(i) + ".npy"
                np.save(descriptorPath, descriptor)


def computeLocalDescriptors(imagesPath, outputPath, descriptor_type, max_num_keypoints, textBoxes = None,
                       backgroundMaskDir = None, multipleImages = "no"):
    """
    This function computes the texture descriptors of the images in the given folder and store them in the output path.

    Parameters
    ----------
    imagesPath : str
        Path of the images.
    outputPath : str
        Path where text descriptors will be saved.
    descriptor_type : str
        Which local descriptor type to use.
    max_num_keypoints: int
        Maximum number of keypoints detections per image.
    textBoxes : list, optional
        Detected text boxes. The default is None.
    backgroundMaskDir : str, optional
        Path where background masks are stored. The default is None.
    multipleImages : str, optional
        Yes if the image can contain more than one paiting. The default is "no".

    Returns
    -------
    None.

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
                    
                    xMin = textBox[0] - xMinP
                    yMin = textBox[1] - yMinP
                    
                    xMax = textBox[2] - xMinP
                    yMax = textBox[3] - yMinP
                    
                    maskNew[yMin:yMax + 1, xMin:xMax + 1] = 0
                 
                    paintingNew[yMin:yMax + 1, xMin:xMax + 1] = [0, 0, 0]
                    
                descriptor = generateLocalDescriptors(descriptor_type, max_num_keypoints, paintingNew)
                
                descriptorPath = outputPath + file[:-4] + "_" + str(i) + ".npy"
                np.save(descriptorPath, descriptor)