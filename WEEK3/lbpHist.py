from skimage import feature
import numpy as np
import cv2


def createLBPhistogram(image, bins, mask, radius = 1):
    """
    This function creates the LBP histogram of the image given. The radius for the LBP and the number
    of bins of the result histogram are given.

    Parameters
    ----------
    image : numpy array (np.uint8)
        image to compute the lbp histogram.
    num_blocs: int
        number of block to divide each image
    radius : int
        radius number to compute lbp.
    bins : int
        number of bins of the histogram per block.

    Returns
    -------
    hist : numpy array 1D
        lbp histogram result of the given image.

    """

    numPoints = 8*radius
    
    # Turn to grayscale image
    imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            
    # Compute lbp
    lbp = feature.local_binary_pattern(imageG, numPoints, radius, method="uniform")
    
    # Compute histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=bins, range=(0, numPoints + 2))
    
    
    # normalize the histogram
    hist = hist.astype("float")
    eps = 1e-7
    hist /= (hist.sum() + eps)
    
    
    return hist
