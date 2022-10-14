import cv2
import numpy as np
import os
from utils.changeColorSpace import changeBGRtoHSV, changeBGRtoYCBCR, changeBGRtoCIELAB, changeBGRtoCIELUV
from multiDimensionalDescriptors import generateMultiDimDescriptors
from hist2Doptimized import Create2DHistogram
from hist3Doptimized import Create3DHistogram



def computeDescriptors(imagesPath, outputPath, colorSpace, numBins, histGenFunc, levels,
                       mask):
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
    mask : numpy array (np.uint8)
        Binary mask to know which pixel has to be used in the process.

    Returns
    -------


    """
    
    
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
            
            descriptor = generateMultiDimDescriptors(image, mask, levels, histGenFunc, numBins)
            
            descriptorPath = outputPath + file[:-4] + ".npy"
            np.save(descriptorPath, descriptor)
            
    
descriptor_dir = "./descriptors/"
color_spaces = ["cielab"]
levels = [0, 1, 2, 3]
bins2D = [10, 45, 90, 130, 180]
bins3D = [5, 10, 20, 30, 40]
    
# Generate the descriptors of BBDD if they are not already generated
for color_space in color_spaces:
    
    # Create folder
    folderName = color_space + "/"
    folderPath = args.descriptor_dir + "BBDD" + "/" + folderName
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        
        computeDescriptors(args.BBDD_dir, folderPath, color_space)
        
        print("BBDD descriptors generated!")
