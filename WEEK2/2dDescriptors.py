# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:27:56 2022

@author: MICHE
"""
import cv2
import math 
import numpy as np
from numpy.linalg import norm
from numpy import dot
import matplotlib.pyplot as plt #importing matplotlib
import os
import sys

def Create2DHistogram(image, bins, colorspaceSize, backgroundMask = None):

    """
    -> Create the 3D histogram of an image
    Parameters :
    - image (np array) : image => e.g. cv2.imread('path_to_image', cv2.COLOR_BGR2RGB) 
    - bins (int) : number of bins
    - colorspaceSize (vector) : vector with the size of each channel => (255,255,255) for RGB
    - backgroundMask (np array) : binary mask
    Return :
    - hisotgram (np array) : 3D histogram
    - numPix (int) : total number of pixels of the image
    kdk
    """
    
    
    # Size of channels
    channelSize1 = colorspaceSize[0]/bins
    channelSize2 = colorspaceSize[1]/bins


    # Create empty histogram
    histogram = np.zeros((bins,bins))

    # Check how much pixels are from foreground
    numPix = 0

     # Check every pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            if backgroundMask[i,j] == 255:

                if(math.floor(image[i,j][0]/channelSize1) == bins):
                    index0 = bins - 1
                else:
                    index0 = math.floor(image[i,j][0]/channelSize1)
                
                if(math.floor(image[i,j][1]/channelSize2) == bins):
                    index1 = bins - 1
                else:
                    index1 = math.floor(image[i,j][1]/channelSize2)

                histogram[index0,index1] += 1
                numPix += 1

    histogram = histogram/numPix

    return histogram