import cv2 
import numpy as np

def changeBGRtoHSV(image):
    """ Function to convert an image from BGR to HSV color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    hsv_image : np.array (uint8)
        Image in HSV color space.

    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Convert every channel to [0-255] range
    hsv_image[:,:,0] = (hsv_image[:,:,0].astype(np.float64) * 255 / 179).astype(np.uint8)
    
    return hsv_image
    
def changeBGRtoYCBCR(image):
    """ Function to convert an image from BGR to YCBCR color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    ycrbc_image : np.array (uint8)
        Image in YCBCR color space.

    """
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    return ycbcr_image  

def changeBGRtoCIELAB(image):
    """ Function to convert an image from BGR to CIELAB color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    ycrbc_image : np.array (uint8)
        Image in CIELAB color space.

    """
    cielab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    return cielab_image  

def changeBGRtoCIELUV(image):
    """ Function to convert an image from BGR to CIELUV color space.
    

    Parameters
    ----------
    image : np.array (uint8)
        Image in BGR color space.

    Returns
    -------
    ycrbc_image : np.array (uint8)
        Image in CIELUV color space.

    """
    cieluv_image = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    
    return cieluv_image