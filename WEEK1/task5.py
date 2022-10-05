import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import os

# METHOD 1
# Obtain the binary mask of an image
def get_binary_mask(image):

    # Converting into graysacle
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding after Gaussian filtering

    # Gaussian filterning is used to remove noises in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Otsu's thresholding gives an arbitrary chosen value as a threshold
    # Image with only two distinct image values (eg: background and painting)
    # where the histogram would only consist of two peaks, 
    # a good threshold would be in the middle of those two values
    ret,binary_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Convert 255 into 1 to have a binary mask
    #binary_mask[binary_mask == 255] = 1 

    #return the binary mask as numpy array
    return binary_mask


# Save the PNG files in q2_result folder

def store_binary_mask():
    pathDataset = "WEEK1/qsd2_w1/"
    pathResultset = "WEEK1/q2_result/"

    # Get file names of the database
    files = os.listdir(pathDataset)

    for file in files:
            # Check if it is an image 
            if file[-4:] == ".jpg" or file[-4:] == ".png":

                #store the image
                image = cv2.imread(pathDataset + file)

                filename = pathResultset + file[:-4] + ".png"

                #save the image in the q2_result folder
                cv2.imwrite(filename, get_binary_mask(image))




















    

