import cv2
import matplotlib.pyplot as plt
import pywt

def createWaveletDescriptor(image):
    """
    This function creates the wavelet descriptor of the diagonal details of the given image

    Parameters
    ----------
    image : numpy array (np.uint8)
        Input image.

    Returns
    -------
    fd : 1D numpy array
        wavelet descriptor the diagonal details of the image

    """
    
    # Change to grayscale
    imageG = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Wavelet transform of image, and plot approximation and details
    wavelets = pywt.dwt2(imageG, 'bior1.3')
    approximation, (horizontalDetails, verticalDetails, diagonalDetails) = wavelets

    # To visualise the transformed image, use the code below
    #plt.imshow(diagonalDetails, interpolation='nearest', cmap=plt.cm.gray)
    
    return diagonalDetails