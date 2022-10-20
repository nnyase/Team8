import cv2
import pywt
import numpy as np

def createWaveletDescriptor(image, bins, waveletType = 'haar'):
    """
    This function creates the wavelet descriptor of the diagonal details of the given image

    Parameters
    ----------
    image : numpy array (np.uint8)
        Input image.
    bins : int
        number of bins of the resulting histogram
    waveletType : str
        type of wavelet used for the transformation

    Returns
    -------
    blockWaveletHistogram : 1D numpy array
        wavelet descriptor the diagonal details of the image

    """
    # Change to grayscale
    imageG = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Wavelet transform of image, the output exists of four seperate images: 
    # approximation, vertical details, horizontal details and diagonal details
    wavedImage = pywt.wavedec2(imageG, waveletType, mode='periodization', level=3)

    # convert the four images into one image to get the wavelet representation of the image
    arr, _ = pywt.coeffs_to_array(wavedImage)

    # create the histogram of the wavelet representation of the block 
    blockWaveletHistogram, _ = np.histogram(arr, bins = bins, range = (0, 255))

    return blockWaveletHistogram