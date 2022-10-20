import cv2
import pywt
import numpy as np

def createWaveletDescriptor(image, k, waveletType = 'haar'):
    """
    This function creates the wavelet descriptor of the diagonal details of the given image

    Parameters
    ----------
    image : numpy array (np.uint8)
        Input image.
    bins : int
        Number of coefficients to take from the image
    waveletType : str
        type of wavelet used for the transformation

    Returns
    -------
    waveletDescriptor : 1D numpy array
        wavelet descriptor

    """
    # Change to grayscale
    imageG = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Wavelet transform of image, the output exists of four seperate images: 
    # approximation, vertical details, horizontal details and diagonal details
    wavedImage = pywt.wavedec2(imageG, waveletType, mode='periodization', level=3)

    # convert the four images into one image to get the wavelet representation of the image
    arr, _ = pywt.coeffs_to_array(wavedImage)

    # create the zigzag of the wavelet representation of the block 
    waveletBlockImageVector = np.concatenate([np.diagonal(arr[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-arr.shape[0], arr.shape[0])])

    waveletDescriptor = waveletBlockImageVector[:k]

    return waveletDescriptor