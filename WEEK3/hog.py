from skimage.feature import hog
import cv2

def createHoGdescriptor(image):
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
    fd, _ = hog(image, feature_vector = True)
    
    return fd

    