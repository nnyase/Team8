import cv2
import numpy as np
import os


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
    
    for file in files:
        # Check if it is an image
        if file[-4:] == ".jpg":
            
            image = cv2.imread(pathDataset + file)
            
            # Here we can change the color space
            
            
            histogram = createHistogram(image)
            
            descriptorPath = outputPath + file[:-4] + ".npy"
            np.save(descriptorPath, histogram)
            
            descriptors[file] = histogram
    
    return descriptors



pathOutputBBDD = "./descriptorsBBDD/"
pathOutputQ1 = "./descriptorsQ1/"
pathOutputQ2 = "./descriptorsQ2/"

computeDescriptors(pathDataset, pathOutputBBDD)
computeDescriptors(pathQuery1, pathOutputQ1)

