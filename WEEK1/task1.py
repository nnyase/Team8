import cv2
import numpy as np
import os
# Manage user input
import sys

# Read dataset images
pathDataset = "../../WEEK1/BBDD/"
pathQuery1 = "../../WEEK1/qsd1_w1/"
pathQuery2 = "../../WEEK1/qsd2_w1/"


# By giving it a grayscale image, returns the histogram related to that image
def calculateHistogram(image):
    # Create empty histogram
    histogram = np.array([0]*256)
    
    # Check every pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            # If the pixel value is not in [0,255] (can be a background pixel) ignore it
            if image[i,j] <= 255 or image[i,j] >= 0:
                
                histogram[image[i,j]] += 1
            

    return histogram

# By giving it a image, returns the concatenation of probability histograms related 
# to that image
# If a image has a background, then background pixels must have values out of [0,255]
# and True value has to be given to the function as the second parameter, 
# not to take into account in the histogram generation
def createHistogram(image, background = False):
    
    # Save concatenation of histograms
    finalProbabilityHistogram = np.array([], dtype = np.float64)
    
    # For each channel get the histogram
    for i in range(image.shape[2]):
        
        # Get the histogram
        if background:
            histogram = calculateHistogram(image[:,:,i])
        else:
            histogram, _  = np.histogram(image[:,:,i], bins = range(0,257))
        
        # Get probability distribution
        histogram = histogram.astype(np.float32) / (image.shape[0]*image.shape[1])
        
        # Concatenate
        finalProbabilityHistogram = np.concatenate((finalProbabilityHistogram,histogram))

    return finalProbabilityHistogram

# By giving it a path to a image, returns a dictionary with every 
# descriptor of images and saves the descriptors in the output folder
def computeDescriptors(imagesPath, outputPath):
    
    descriptors = {}
    
    # Get file names of the database
    files = os.listdir(pathDataset)

    # Asking the user if he wants to convert image into HSV colorspace
    answer = query_yes_no("Do you want to convert image into HSV mode?")
    
    for file in files:
        # Check if it is an image
        if file[-4:] == ".jpg":
            
            image = cv2.imread(pathDataset + file)
            
            # Here we can change the color space

            #convert the image into HSV
            if answer == True:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            histogram = createHistogram(image)
            
            descriptorPath = outputPath + file[:-4] + ".npy"
            np.save(descriptorPath, histogram)
            
            descriptors[file] = histogram
    
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


pathOutputBBDD = "./descriptorsBBDD/"
pathOutputQ1 = "./descriptorsQ1/"
pathOutputQ2 = "./descriptorsQ2/"

# computeDescriptors(pathDataset, pathOutputBBDD)
# computeDescriptors(pathQuery1, pathOutputQ1)

