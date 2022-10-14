import numpy as np

def Create3DHistogram(image, bins, colorSpaceSize, mask):
    """
    -> Create the 3D histogram of an image

    Parameters :

    - image (np array) : image => e.g. cv2.imread('path_to_image', cv2.COLOR_BGR2RGB) 
    - bins (int) : number of bins
    - colorSpaceSize (vector) : vector with the size of each channel => (255,255,255) for RGB
    - mask (np array) : binary mask

    Return :

    - flatHist (np array) : flattened 3D histogram
    
    """
    newImage = image.copy()
    
    # Set 0,0,0 to mask pixels
    newImage[mask == 255] = [0,0,0]
    numBackgroundPixels = np.sum(mask == 255)
    
    # Get channels
    channel1 = image[:, :, 0].reshape(-1)
    channel2 = image[:, :, 1].reshape(-1)
    channel3 = image[:, :, 2].reshape(-1)
    
    
    # Compute histogram
    hist, _ = np.histogramdd((channel1, channel2, channel3), bins = bins, range = [(0, colorSpaceSize[0]), (0,colorSpaceSize[1]), (0,colorSpaceSize[2])])
    
    # Remove the values of the mask pixels
    hist[0,0,0] -= numBackgroundPixels
    
    # Every pixels is in mask
    if np.sum(hist) == 0:
        hist[:,:,:] = [1,1,1]
        
    # Normalize
    hist /= np.sum(hist)    
    
    # Flat
    flatHist = hist.reshape(-1)
    
    return flatHist