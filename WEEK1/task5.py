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


# METHOD 2
# 1. Convert our image into Greyscale
# 2. Perform simple thresholding to build a mask for the foreground and background
# 3. Determine the foreground and background based on the mask
# 4. Reconstruct original image by combining foreground and background

def get_binary_mask2(myimage):
    # First Convert to Grayscale
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    # Trunc thresholding - Truncate Thresholding ( THRESH_TRUNC ) ·
    #The destination pixel is set to the threshold ( thresh ), if the source pixel value is greater than the threshold.
    ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
    # Threshold to remove background from grayscaled image
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
    # Threshold to extract the background from the grayscaled image
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Combine the background and foreground to obtain our final image with background extracted
    finalimage = background+foreground
    bg = cv2.bitwise_not(background)
    return bg

#METHOD 3
#Convert our image into HSV color space
#Perform simple thresholding to create a map using Numpy based on Saturation and Value
#Combine the map from S and V into a final mask
#Determine the foreground and background based on the combined mask
#Reconstruct original image by combining extracted foreground and background

def get_binary_mask3(myimage):
    
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
    binary_mask = cv2.bitwise_not(background)
 
    return binary_mask



    

