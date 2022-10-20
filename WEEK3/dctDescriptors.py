import cv2
import numpy as np


def createDCTdescritor(image, k, mask):
    """
    This function creates the DCT descriptors of the input image

    Parameters
    ----------
    image : numpy array (np.uint8)
        Image to generate descriptors.
    num_blocks : int
        Number of block to divide the image.
    k : int
        Number of coefficients to take from each block.

    Returns
    -------
    resultHist : 1D numpy array
        Image descriptors.

    """
    # Turn image to grayscale
    imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # Convert to float
    imageG = imageG.astype(np.float32)/255.
    
    
    # Calculate dct
    dctBlockImage = cv2.dct(imageG)
    # Get zig-zag
    dctBlockImageVector = np.concatenate([np.diagonal(dctBlockImage[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctBlockImage.shape[0], dctBlockImage.shape[0])])
    
    # Get first k elements
    blockDescriptor = dctBlockImageVector[:k]
            
            
            
    return blockDescriptor
    