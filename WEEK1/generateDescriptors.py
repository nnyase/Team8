import cv2
import numpy as np
import os


def changeBGRtoHSV(image):
    """ Function to convert an image from BGR to HSV color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    hsv_image : np.array (uint8)
        Image in HSV color space.

    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Convert every channel to [0-255] range
    hsv_image[:,:,0] = (hsv_image[:,:,0].astype(np.float64) * 255 / 179).astype(np.uint8)
    
    return hsv_image
    
def changeBGRtoYCBCR(image):
    """ Function to convert an image from BGR to YCBCR color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    ycrbc_image : np.array (uint8)
        Image in YCBCR color space.

    """
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    return ycbcr_image  

def changeBGRtoCIELAB(image):
    """ Function to convert an image from BGR to CIELAB color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    ycrbc_image : np.array (uint8)
        Image in CIELAB color space.

    """
    cielab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    return cielab_image  

def changeBGRtoCIELUV(image):
    """ Function to convert an image from BGR to CIELUV color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    ycrbc_image : np.array (uint8)
        Image in CIELUV color space.

    """
    cieluv_image = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    
    return cieluv_image

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
            if backgroundMask[i,j] == 1:
                
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
    
    # Create output folder
    resultsPath = outputPath + "descriptors_" + imagesPath.split('/')[-2] + "_" + colorSpace + "/"  
    os.mkdir(resultsPath)
    
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
                binaryMask = cv2.imread(backgroundMaskDir + file[:-4] + ".png")
                descriptor = generateDescriptor(image, background, binaryMask)
            else:
                descriptor = generateDescriptor(image)
            
            descriptorPath = resultsPath + file[:-4] + ".npy"
            np.save(descriptorPath, descriptor)
            
            descriptors[file] = descriptor
    
    return descriptors

if __name__ == "__main__":

    input_dir = "../../WEEK1/qsd1_w1/"
    output_dir = "./descriptors/"
    color_space = "hsv"
    
    mask_dir = 'None'
    
    if mask_dir != 'None':
        computeDescriptors(input_dir, output_dir, color_space, True, mask_dir)
    else:
        computeDescriptors(input_dir, output_dir, color_space)
        

    

