import numpy as np

def Create3DHistogram(image, bins, mask):
    """
    -> Create the 3D histogram of an image

    Parameters :

    - image (np array) : image => e.g. cv2.imread('path_to_image', cv2.COLOR_BGR2RGB) 
    - bins (int) : number of bins
    - mask (np array) : binary mask

    Return :

    - flatHist (np array) : flattened 3D histogram
    
    """
    newImage = image.copy()
    
    # Set 0,0,0 to mask pixels
    newImage[mask == 0] = [0,0,0]
    numBackgroundPixels = np.sum(mask == 0)
    
    # Get channels
    channel1 = newImage[:, :, 0].reshape(-1)
    channel2 = newImage[:, :, 1].reshape(-1)
    channel3 = newImage[:, :, 2].reshape(-1)
    
    
    # Compute histogram
    hist, _ = np.histogramdd((channel1, channel2, channel3), bins = bins, range = [(0, 255), (0,255), (0,255)])
    
    # Remove the values of the mask pixels
    hist[0,0,0] -= numBackgroundPixels
    
    # Every pixels is in mask
    if np.sum(hist) == 0:
        hist[:,:,:] = 1
        
    # Normalize
    hist /= np.sum(hist)    
    
    # Flat
    flatHist = hist.reshape(-1)
    
    return flatHist