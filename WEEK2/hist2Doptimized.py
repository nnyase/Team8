import numpy as np

def Create2DHistogram(image, bins, mask):
    """
    -> Create the 2D histogram of an image

    Parameters :

    - image (np array) : image => e.g. cv2.imread('path_to_image', cv2.COLOR_BGR2RGB) 
    - bins (int) : number of bins
    - mask (np array) : binary mask

    Return :

    - flatHist (np array) : flattened 2D histogram
    
    """
    newImage = image.copy()
    
    # Set 0,0,0 to mask pixels
    newImage[mask == 0] = [0,0,0]
    numBackgroundPixels = np.sum(mask == 0)
    
    # Get channels
    channel1 = newImage[:, :, 0].reshape(-1)
    channel2 = newImage[:, :, 1].reshape(-1)
    channel3 = newImage[:, :, 2].reshape(-1)
    
    
    # Channel1 and channel2
    # Compute histogram
    hist, _ = np.histogramdd((channel1, channel2), bins = bins, range = [(0, 255), (0,255)])
    
    # Remove the values of the mask pixels
    hist[0,0] -= numBackgroundPixels
    
    # Flat
    flatHist1 = hist.reshape(-1)
    
    # Channel2 and channel3
    # Compute histogram
    hist, _ = np.histogramdd((channel2, channel3), bins = bins, range = [(0, 255), (0,255)])
    
    # Remove the values of the mask pixels
    hist[0,0] -= numBackgroundPixels 
    
    # Flat
    flatHist2 = hist.reshape(-1)
    
    # Channel1 and channel3
    # Compute histogram
    hist, _ = np.histogramdd((channel1, channel3), bins = bins, range = [(0, 255), (0,255)])
    
    # Remove the values of the mask pixels
    hist[0,0] -= numBackgroundPixels   
    
    # Flat
    flatHist3 = hist.reshape(-1)
    
    # Concatenate
    resultHist = np.concatenate([flatHist1, flatHist2, flatHist3])
    
    # Every pixels is in mask
    if np.sum(resultHist) == 0:
        resultHist[:] = 1
    
    # Normalize
    resultHist /= np.sum(resultHist)
    
    return resultHist
