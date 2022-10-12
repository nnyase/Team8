import numpy as np
import math 

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

    return histogram, numPix


def print3DHistogram(histogram):

    """
    -> Print the 3D histogram like [x,y,z] = % of occurence in the image
    
    Parameters :

    - histogram (np array) 

    """
    for x in range(histogram.shape[0]):
        for y in range(histogram.shape[1]):
            for z in range(histogram.shape[2]):

                print("[" + str(x) + "," + str(y) + "," + str(z) + "] = " +  str(histogram[x,y,z]) + "\n")


#Compute distance

#EuclidianDistance
def EuclidianDistance(hist1,hist2):
    """
    -> Return the Euclidian distance between two 3D histogram
    """
    return (np.sum((hist1 - hist2)**2))**0.5

#L1 distance
def L1_Distance(hist1,hist2):
    """
    -> Return the L1 distance between two 3D histogram
    """
    return (np.sum(abs(hist1 - hist2)))

#X2 distance
def X2_Distance(hist1,hist2):
    """
    -> Return the X2 distance between two 3D histogram
    """
    return np.sum(np.divide((hist1 - hist2)**2, hist1 + hist2))

#Hellinger kernel 
def Hellinger_distance(hist1,hist2):
    """
    -> Return the Hellinger distance between two 3D histogram
    """
    return np.sum((np.multiply(hist1,hist2))**2)
 



