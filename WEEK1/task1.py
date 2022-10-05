import cv2
import numpy as np
import os
# Manage user input
import sys


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
    
def changeBGRtoYCRBC(image):
    """ Function to convert an image from BGR to YCRBC color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    ycrbc_image : np.array (uint8)
        Image in YCRBC color space.

    """
    ycrbc_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    return ycrbc_image  

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

    """
    
    # Create empty histogram
    histogram = np.array([0]*256)
    
    # Check every pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            # If the pixel is not part of the background take it into account
            if backgroundMask[i,j] == 1:
                
                histogram[image[i,j]] += 1
            

    return histogram


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
            histogram = genHistoNoBackground(image[:,:,i], backgroundMask)
        else:
            histogram, _  = np.histogram(image[:,:,i], bins = range(0,257))
        
        # Get probability distribution
        histogram = histogram.astype(np.float32) / (image.shape[0]*image.shape[1])
        
        # Concatenate
        finalProbabilityHistogram = np.concatenate((finalProbabilityHistogram,histogram))

    return finalProbabilityHistogram


def computeDescriptors(imagesPath, outputPath, colorSpace):
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
        rgb, hsv, cielab, cieluv, ycrbc are the options.

    Returns
    -------
    descriptors : dictionary
        A dictionary that contains the image file name and its descriptors.

    """
    
    
    descriptors = {}
    
    # Get file names of the database
    files = os.listdir(pathDataset)
    
    for file in files:
        # Check if it is an image
        if file[-4:] == ".jpg":
            
            image = cv2.imread(pathDataset + file)

            # Convert the image into the new color space
            if colorSpace == "hsv":
                image = changeBGRtoHSV(image)
            elif colorSpace == "cielab":
                image = changeBGRtoCIELAB(image)
            elif colorSpace == "cieluv":
                image = changeBGRtoCIELUV(image)
            elif colorSpace == "ycrbc":
                image = changeBGRtoYCRBC(image)
            
            descriptor = generateDescriptor(image)
            
            descriptorPath = outputPath + file[:-4] + ".npy"
            np.save(descriptorPath, descriptor)
            
            descriptors[file] = descriptor
    
    return descriptors

# This function is used to get the answer of the user, "yes" or "no" and avoiding currents typing errors
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


# Read dataset images
pathDataset = "../../WEEK1/BBDD/"
pathQuery1 = "../../WEEK1/qsd1_w1/"
pathQuery2 = "../../WEEK1/qsd2_w1/"

# Output descriptors
pathOutputBBDD = "./descriptorsBBDD/"
pathOutputQ1 = "./descriptorsQ1/"
pathOutputQ2 = "./descriptorsQ2/"


colorSpace = "cielab"
computeDescriptors(pathDataset, pathOutputBBDD, colorSpace)
# computeDescriptors(pathQuery1, pathOutputQ1)

