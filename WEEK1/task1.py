import cv2
import numpy as np
import os


# Read dataset images
pathDataset = "../BBDD/"
pathQuery1 = "../qsd1_w1/"
pathQuery2 = "../qsd2_w1/"


# By giving it a image, returns the concatenation of probability histograms related 
# to that image
def createHistogram(image):
    
    # Save concatenation of histograms
    finalProbabilityHistogram = np.array([], dtype = np.float64)
    
    # For each channel get the histogram
    for i in range(image.shape[2]):
        
        # Values from 0 to 255
        histogram, bin_edges = np.histogram(image[:,:,i], bins=range(0,257))
        
        # Get probability distribution
        histogram = histogram.astype(np.float32) / (image.shape[0]*image.shape[1])
        
        # Concatenate
        finalProbabilityHistogram = np.concatenate((finalProbabilityHistogram,histogram))

    return finalProbabilityHistogram

# By giving it the path dataset, returns a library with every 
# descriptor of the BBDD
def computeBBDDdescriptors(pathDataset):
    
    descriptors = {}
    
    # Get file names of the database
    files = os.listdir(pathDataset)
    
    for file in files:
        # Check if it is an image
        if file[-4:] == ".jpg":
            
            image = cv2.imread(pathDataset + file)
            
            # Here we can change the color space
            
            
            histogram = createHistogram(image)
            descriptors[file] = histogram
    
    return descriptors




computeBBDDdescriptors(pathDataset)



