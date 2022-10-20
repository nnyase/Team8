import numpy as np

def generateMultiResDescriptors(image, mask, levels, histGenFunc, bins):
    """
    This function generates the multidimensional descriptors computing the Spatial Pyramin Matching.

    Parameters
    ----------
    image : numpy array (uint8)
        input image.
    mask : numpy array (uint8)
        binary mask with 0 values in pixels that we have not to take into account in the generation.
    levels : int
        number of level. 0 == not creating blocks
    histGenFunc : function
        function to generate the histograms giving the image, mask and number of bins.
    bins : int
        number of bins in each histogram.

    Returns
    -------
    resultHistogram : numpy array
        multidimensional descriptor generated.

    """
    resultHistogram = np.array([])
    
    for i in range(levels + 1):
        
        # Compute this level weights
        if i == 0 or i == 1:
            weight = 1/2**levels
        else:
            weight = 2**(i - levels - 1)
        
        levelHistogram = np.array([])
        
        # Compute histograms
        num = 2**i
        difW = int(image.shape[1]/num)
        remainingW = image.shape[1] % num
        difH = int(image.shape[0]/num)
        remainingH = image.shape[0] % num
        
        
        actualH = 0
        
        
        # Compute every roi
        for h in range(num):
            # Compute roi height
            if remainingH > 0:
                roiH = difH + 1
                remainingH -= 1
            else:
                roiH = difH
            
            actualW = 0
            remainingW = image.shape[1] % num
            
            for w in range(num): 
                # Compute roi width
                if remainingW > 0:
                    roiW = difW + 1
                    remainingW -= 1
                else:
                    roiW = difW
            
                
                hist = histGenFunc(image[actualH: actualH + roiH, actualW: actualW + roiW], bins, mask[actualH: actualH + roiH, actualW: actualW + roiW])
                
                actualW = actualW + roiW
                
                levelHistogram = np.concatenate([levelHistogram, hist])
            
            actualH = actualH + roiH 
        
        levelHistogram = levelHistogram / np.sum(levelHistogram)
        resultHistogram = np.concatenate([resultHistogram, levelHistogram * weight])
    
    return resultHistogram
