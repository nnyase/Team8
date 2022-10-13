# -*- coding: utf-8 -*-
import cv2
import math 
import numpy as np
from numpy.linalg import norm
from numpy import dot
import matplotlib.pyplot as plt #importing matplotlib
import os
import sys

""" The goal of this script is to generate a descriptor 
    folder with the following options
    1.- Generate 2 level descriptors (each image divided in 4 subimages)
    2.- Generate 3 level descriptors (each image divided in 16 subimages)
    3.- Generate 2d histograms (Only 1 Level for 2d histograms avalaibable )
    """


def concatenatedHistogram(roi):
    """ This function generates a concatenated histogram of a 3 channels image
    Parameters
    roi: it could be an image or a subimage(ROI)
    ----------
    Returns
    -------
    con:  3 histograms concatenated
    """
    histCh1 = (cv2.calcHist([roi],[0],None,[256],[0,256]))/(roi.shape[0]*roi.shape[1])
    histCh2 = cv2.calcHist([roi],[1],None,[256],[0,256])/(roi.shape[0]*roi.shape[1])
    histCh3 = cv2.calcHist([roi],[2],None,[256],[0,256])/(roi.shape[0]*roi.shape[1])
    con = np.concatenate((histCh1, histCh2,histCh3))
    return con


def toyHistGenFunc(bins, image, mask):
    
    result = np.zeros(bins)
    result[0] = 1
    
    return result



toyImage = np.zeros([587,678,3])
toyMask = np.zeros([587,678])
multiDimensionalDescriptors(toyImage, toyMask, 5, toyHistGenFunc, 6)

def getGrid2Level(M,N):     
    
    """ This function generates a vector of a vector of 2 points to be used
        to split an image in 4 parts, the vector has the coordinates
        of 4 rois which aren't overlaped
    Parameters
    M: Cols of the image
    N: Rows of the image
    ----------
    Returns
    pointVector: vector with 4 rois coordinates 
    
    -------
    """
    grid2D =  [[0,0],                   #A
              [int((N/2)),0],           #B
              [N,0],                    #C
              [0,int((M/2))],           #D
              [int((N/2)),int((M/2))],  #E
              [int(N),int((M/2))],      #F
              [0,int(M)],               #G
              [int((N/2)),int(M)],      #H
              [N,M]]                    #I

    ROI1 = [grid2D[0],[grid2D[4][0]-1,grid2D[4][1]-1]]
    ROI2 = [grid2D[1],[grid2D[5][0],grid2D[5][1]-1]]
    ROI3 = [grid2D[3],[grid2D[7][0]-1,grid2D[7][1]]]
    ROI4 = [grid2D[4],grid2D[8]]
    pointVector=[ROI1,ROI2,ROI3,ROI4]
           
    return pointVector

    
def getDescriptors2Level(img_src,level,colorSpace):
    """ This function:
        1.- Gets an image
        2.- Uses get2GridLevel function to split the image in 4 subimages
        3.- Generate the concatenated 3 channel histogram of each subimage
        4.- Concatenate the concatenated histograms of every subimage 
    Parameters
    img_src: Main image
    colorSpace: color space to be used 
    ----------
    Returns
    conFinal: 4 concatenated  3 channels histograms 
    
    -------
    """
    #img_src =  cv2.imread(imagePath,1)
    if colorSpace==1:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
    elif colorSpace==2:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    elif colorSpace==3:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2LAB)
    elif colorSpace==4:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2YCrCb)
    else:
        print("BGR color space selected")
        
        

    roiVector=getGrid2Level(img.shape[0],img.shape[1])

    roi1 = img[roiVector[0][0][0]:roiVector[0][1][1], 
                     roiVector[0][0][1]:roiVector[0][1][0]]
    roi2 = img[roiVector[1][0][1]:roiVector[1][1][1], 
                     roiVector[1][0][0]:roiVector[1][1][0]]
    roi3 = img[roiVector[2][0][1]:roiVector[2][1][1], 
                     roiVector[2][0][0]:roiVector[2][1][0]]
    roi4 = img[roiVector[3][0][1]:roiVector[3][1][1], 
                     roiVector[3][0][0]:roiVector[3][1][0]]


    con1 = concatenatedHistogram(roi1)
    con2 = concatenatedHistogram(roi2)
    con3 = concatenatedHistogram(roi3)
    con4 = concatenatedHistogram(roi4)
       
    conFinal = np.concatenate((con1,con2,con3,con4))

    return conFinal


def getDescriptors3Level(img_src,level,colorSpace):
    #img_src =  cv2.imread(imagePath,1)
    if colorSpace==1:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
    elif colorSpace==2:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    elif colorSpace==3:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2LAB)
    elif colorSpace==4:
        img = cv2.cvtColor(img_src,cv2.COLOR_BGR2YCrCb)
    else:
        print("BGR color space selected")
        
        

    roiVector=getGrid2Level(img.shape[0],img.shape[1])
    #Rectangle marker
    # rect_img = img[roiVector[0][0][0]:roiVector[0][0][1], roiVector[0][1][0]:roiVector[0][1][1]]
    roi1 = img[roiVector[0][0][0]:roiVector[0][1][1], 
                     roiVector[0][0][1]:roiVector[0][1][0]]
    roi2 = img[roiVector[1][0][1]:roiVector[1][1][1], 
                     roiVector[1][0][0]:roiVector[1][1][0]]
    roi3 = img[roiVector[2][0][1]:roiVector[2][1][1], 
                     roiVector[2][0][0]:roiVector[2][1][0]]
    roi4 = img[roiVector[3][0][1]:roiVector[3][1][1], 
                     roiVector[3][0][0]:roiVector[3][1][0]]

## Roi1- subRoi A,B,C,D
    roiVector=getGrid2Level(roi1.shape[0],roi1.shape[1])
    
    roi1_A = roi1[roiVector[0][0][0]:roiVector[0][1][1], 
                     roiVector[0][0][1]:roiVector[0][1][0]]
    roi1_B = roi1[roiVector[1][0][1]:roiVector[1][1][1], 
                     roiVector[1][0][0]:roiVector[1][1][0]]
    roi1_C = roi1[roiVector[2][0][1]:roiVector[2][1][1], 
                     roiVector[2][0][0]:roiVector[2][1][0]]
    roi1_D = roi1[roiVector[3][0][1]:roiVector[3][1][1], 
                     roiVector[3][0][0]:roiVector[3][1][0]]
   
    con1_A = concatenatedHistogram(roi1_A)
    con1_B = concatenatedHistogram(roi1_B)
    con1_C = concatenatedHistogram(roi1_C)
    con1_D = concatenatedHistogram(roi1_D)
    con1   = np.concatenate((con1_A, con1_B,con1_C,con1_D))
    
## Roi2- subRoi A,B,C,D

    roiVector=getGrid2Level(roi2.shape[0],roi2.shape[1])
    roi2_A = roi2[roiVector[0][0][0]:roiVector[0][1][1], 
                     roiVector[0][0][1]:roiVector[0][1][0]]
    roi2_B = roi2[roiVector[1][0][1]:roiVector[1][1][1], 
                     roiVector[1][0][0]:roiVector[1][1][0]]
    roi2_C = roi2[roiVector[2][0][1]:roiVector[2][1][1], 
                     roiVector[2][0][0]:roiVector[2][1][0]]
    roi2_D = roi2[roiVector[3][0][1]:roiVector[3][1][1], 
                     roiVector[3][0][0]:roiVector[3][1][0]]

    con2_A = concatenatedHistogram(roi2_A)
    con2_B = concatenatedHistogram(roi2_B)
    con2_C = concatenatedHistogram(roi2_C)
    con2_D = concatenatedHistogram(roi2_D)
    
    con2   = np.concatenate((con2_A, con2_B,con2_C,con2_D))
      
 ## Roi3- subRoi A,B,C,D

    roiVector=getGrid2Level(roi3.shape[0],roi3.shape[1])
    roi3_A = roi3[roiVector[0][0][0]:roiVector[0][1][1], 
                     roiVector[0][0][1]:roiVector[0][1][0]]
    roi3_B = roi3[roiVector[1][0][1]:roiVector[1][1][1], 
                     roiVector[1][0][0]:roiVector[1][1][0]]
    roi3_C = roi3[roiVector[2][0][1]:roiVector[2][1][1], 
                     roiVector[2][0][0]:roiVector[2][1][0]]
    roi3_D = roi3[roiVector[3][0][1]:roiVector[3][1][1], 
                     roiVector[3][0][0]:roiVector[3][1][0]]
    con3_A = concatenatedHistogram(roi3_A)
    con3_B = concatenatedHistogram(roi3_B)
    con3_C = concatenatedHistogram(roi3_C)
    con3_D = concatenatedHistogram(roi3_D)
    
    con3   = np.concatenate((con3_A, con3_B,con3_C,con3_D))
            
 ## Roi4- subRoi A,B,C,D

    roiVector=getGrid2Level(roi4.shape[0],roi4.shape[1])
    roi4_A = roi4[roiVector[0][0][0]:roiVector[0][1][1], 
                     roiVector[0][0][1]:roiVector[0][1][0]]
    roi4_B = roi4[roiVector[1][0][1]:roiVector[1][1][1], 
                     roiVector[1][0][0]:roiVector[1][1][0]]
    roi4_C = roi4[roiVector[2][0][1]:roiVector[2][1][1], 
                     roiVector[2][0][0]:roiVector[2][1][0]]
    roi4_D = roi4[roiVector[3][0][1]:roiVector[3][1][1], 
                     roiVector[3][0][0]:roiVector[3][1][0]]
    con4_A = concatenatedHistogram(roi4_A)
    con4_B = concatenatedHistogram(roi4_B)
    con4_C = concatenatedHistogram(roi4_C)
    con4_D = concatenatedHistogram(roi4_D)
    
    con4   = np.concatenate((con4_A, con4_B,con4_C,con4_D)) 
    
    con_final= np.concatenate((con1, con2,con3,con4))   
    # cv2.imshow("roi1_A", roi1_A)
    # cv2.imshow("roi1_B", roi1_B)
    # cv2.imshow("roi1_C", roi1_C)
    # cv2.imshow("roi1_D", roi1_D)
    # cv2.imshow("roi1", roi1)
    # cv2.waitKey(0)
    return con_final

def Create3DHistogram(image, bins, colorspaceSize, backgroundMask = None):

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
    channelSize3 = colorspaceSize[2]/bins

    # Check if the size of the channel is divided by bins
    if(colorspaceSize[0]%bins != 0 or colorspaceSize[1]%bins != 0  or colorspaceSize[2]%bins != 0):
        print("Size of image channels aren't divisible by bins \n")
        return 0

    # Create empty histogram
    histogram = np.zeros((bins,bins,bins))

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

                if(math.floor(image[i,j][2]/channelSize3) == bins):
                    index2 = bins - 1
                else:
                    index2 = math.floor(image[i,j][2]/channelSize3)

                histogram[index0,index1,index2] += 1
                numPix += 1

    histogram = histogram/numPix

    return histogram
def computeDescriptors(imagesPath, outputPath, colorSpace, getDescriptorSelector , background = False, 
                       backgroundMaskDir = None):
    """ This function generates and fill a descriptor folder with the following options
        1.- 2 Level descriptors
        2.- 3 level descriptors
        3.- 2d histrogram descriptors (Only for level 1 )
    Parameters
    imagePath:
    outputPath:
    colorSpace: only used to generate the path dir the colorSpace 
                is currently selectecd in  any of the getDescritor
                functions (getDescriptors3Level, getDescriptors2Level, getDescriptors2Dhist ...)
    getDescriptorSelector : 
                                    1----> 2 Level descriptors, 
                                    2----> 3 level descriptors 
                                    3----> 2d histrogram descriptors (Only for level 1 )
    background:
    backgroundMaskDir
    ----------
    Returns
    -------
    con:  3 histograms concatenated
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
            if getDescriptorSelector==1:
                descriptor = getDescriptors2Level(image,1,1)
            elif getDescriptorSelector==2:
                descriptor = getDescriptors3Level(image,1,1)
            elif getDescriptorSelector==3:
                 colorspaceSize=(255,255,255)
                 descriptor = Create3DHistogram(image, 3, colorspaceSize,np.zeros(image.shape[:-1],dtype=np.uint8)+255)

                
            descriptorPath = resultsPath + file[:-4] + ".npy"
            np.save(descriptorPath, descriptor)
            
            descriptors[file] = descriptor
    
    return descriptors
#getDescriptors3Level(pathDdbb,1,3)

input_dir = "./BBDD/"
output_dir = "./Descriptors/3d/"
color_space = "bgr"

# test
pathDdbb = "./BBDD/bbdd_00000.jpg"
descriptorPath2 = "./Descriptors/Level2/"
descriptorPath3 = "./Descriptors/Level3/"      

computeDescriptors(input_dir, output_dir,color_space,3)
#getDescriptors3Level(pathDdbb,1,3)  










