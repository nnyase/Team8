from skimage.feature import hog
import cv2
import numpy as np

def createHoGdescriptor(image, orientations, mask):
    """
    This function creates the HoG descriptor of the given image

    Parameters
    ----------
    image : numpy array (np.uint8)
        Input image.

    Returns
    -------
    fd : 1D numpy array
        HoG descriptor of the image.

    """
    
    # Change to grayscale
    imageG = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Generate hog
    pixels_per_block = (image.shape[0], image.shape[1])
    fd = hog(imageG, orientations, cells_per_block=(1, 1), pixels_per_cell = pixels_per_block, feature_vector = True)
    
    resultHist = fd.astype("float")
    eps = 1e-7
    resultHist /= (resultHist.sum() + eps)
    
    return resultHist

    