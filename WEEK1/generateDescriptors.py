import cv2
import numpy as np
import os
from utils.changeColorSpace import changeBGRtoHSV, changeBGRtoYCBCR, changeBGRtoCIELAB, changeBGRtoCIELUV

def genHistoNoBackground(image, backgroundMask):
    """ This function generates the histogram of a one channel image, not
    taking into account the backgorund pixels.
    

    Parameters
    ----------
    image : numpy array (uint8)
        One channel image.
    backgroundMask: numpy array (uint8)
        Binary mask of the background of the image.
        
    Returns
    -------
    histogram : numpy array (int)
        The histogram related to the values of the input image.
    numPix : int
        The number of pixels that are from the foreground.

    """
    
    # Create empty histogram
    histogram = np.array([0]*256)
    
    # Check how much pixels are from foreground
    numPix = 0
    
    # Check every pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            # If the pixel is not part of the background take it into account
            if backgroundMask[i,j] == 255:
                
                histogram[image[i,j]] += 1
                numPix += 1
            

    return histogram, numPix


def generateDescriptor(image, background = False, backgroundMask = None):
    """ This function creates the descriptor of a three channel image. 
    

    Parameters
    ----------
    image : numpy array (uint8)
        The image from which the descriptor will be generated.

    background : boolean, optional
        Parameter to state if the image has background related pixels,
        not to take them into account in the decriptor generation process. 
        The default value is False.
    
    backgroundMask : numpy array (uint8), optional
        A binary mask image of the background.
        

    Returns
    -------
    finalProbabilityHistogram : numpy array (float)
        Descriptor of the image.

    """
    
    # Save concatenation of histograms
    finalProbabilityHistogram = np.array([], dtype = np.float64)
    
    # For each channel get the histogram
    for i in range(image.shape[2]):
        
        # Get the histogram
        if background:
            histogram, numPixels = genHistoNoBackground(image[:,:,i], backgroundMask)
            # Get probability distribution
            histogram = histogram.astype(np.float32) / numPixels
        else:
            histogram, _  = np.histogram(image[:,:,i], bins = range(0,257))
            # Get probability distribution
            histogram = histogram.astype(np.float32) / (image.shape[0]*image.shape[1])
        
        
        # Concatenate
        finalProbabilityHistogram = np.concatenate((finalProbabilityHistogram,histogram))

    return finalProbabilityHistogram


def computeDescriptors(imagesPath, outputPath, colorSpace, background = False, 
                       backgroundMaskDir = None ):
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
    background : boolean, optional
        Parameter to state if the image has background related pixels,
        not to take them into account in the decriptor generation process. 
        The default value is False.
    backgroundMaskDir : string, optional
        Directory where binary masks of the background of images are.

    Returns
    -------
    descriptors : dictionary
        A dictionary that contains the image file name and its descriptors.

    """
    
    
    descriptors = {}
    
    # Get file names of the database
    files = os.listdir(imagesPath)
    
    for file in files:
        # Check if it is an image
        if file[-4:] == ".jpg":
            
            image = cv2.imread(imagesPath + file)

            # Convert the image into the new color space
            if colorSpace == "hsv":
                image = changeBGRtoHSV(image)
            elif colorSpace == "cielab":
                image = changeBGRtoCIELAB(image)
            elif colorSpace == "cieluv":
                image = changeBGRtoCIELUV(image)
            elif colorSpace == "ycbcr":
                image = changeBGRtoYCBCR(image)
            
            if background:
                # Read binary mask
                binaryMask = cv2.imread(backgroundMaskDir + file[:-4] + ".png", cv2.IMREAD_GRAYSCALE)
                descriptor = generateDescriptor(image, background, binaryMask)
            else:
                descriptor = generateDescriptor(image)
            
            descriptorPath = outputPath + file[:-4] + ".npy"
            np.save(descriptorPath, descriptor)
            
            descriptors[file] = descriptor
    
    return descriptors

    

