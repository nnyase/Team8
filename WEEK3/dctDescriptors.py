import cv2
import numpy as np


def createDCTdescritor(image, num_blocks, k):
    """
    This function creates the DCT descriptors of the input image

    Parameters
    ----------
    image : numpy array (np.uint8)
        Image to generate descriptors.
    num_blocks : int
        Number of block to divide the image.
    k : TYPE
        Number of coefficients to take from each block.

    Returns
    -------
    resultHist : 1D numpy array
        Image descriptors.

    """
    # Turn image to grayscale
    imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Init result
    resultHist = []
    
            
    # Compute histograms
    difW = int(image.shape[1]/num_blocks)
    remainingW = image.shape[1] % num_blocks
    difH = int(image.shape[0]/num_blocks)
    remainingH = image.shape[0] % num_blocks
    
    
    actualH = 0
    
    
    # Compute every block
    for h in range(num_blocks):
        # Compute roi height
        if remainingH > 0:
            roiH = difH + 1
            remainingH -= 1
        else:
            roiH = difH
        
        actualW = 0
        remainingW = image.shape[1] % num_blocks
        
        for w in range(num_blocks): 
            # Compute roi width
            if remainingW > 0:
                roiW = difW + 1
                remainingW -= 1
            else:
                roiW = difW
        
            
            blockImage = imageG[actualH: actualH + roiH, actualW: actualW + roiW]
            # Convert to float
            blockImage = blockImage.astype(np.float32)/255.
            
            
            # Calculate dct
            dctBlockImage = cv2.dct(blockImage)
            # Get zig-zag
            dctBlockImageVector = np.concatenate([np.diagonal(dctBlockImage[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctBlockImage.shape[0], dctBlockImage.shape[0])])
            
            # Get first k elements
            blockDescriptor = dctBlockImageVector[:k]
            
            
            # Concatenate
            resultHist = np.concatenate([resultHist, blockDescriptor])
            
            actualW = actualW + roiW
        
        actualH = actualH + roiH 
            
    return resultHist
    