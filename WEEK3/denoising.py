import cv2
from skimage.restoration import estimate_sigma, denoise_tv_bregman
import numpy as np
import os

def denoising(imagePath):

    """
    This function denoise the image if the image contains noises

    Parameters
    ----------
    imagePath : string
        Path of the input image.
    
    Returns
    -------
    newImage : 2D numpy array
        Ouput image denoised.

    """

    img = cv2.imread(imagePath)
    
    # Documentation : https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.estimate_sigma
    # Check if the image contains noises

    if estimate_sigma(img, channel_axis=-1, average_sigmas=True) >= 2:
        
        # Differents filter that I have already used
        #newImage = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)
        #newImage = denoise_tv_bregman(img, weight=5.0, max_num_iter=100, eps=0.001, isotropic=True, channel_axis=-1)
        #newImage = denoise_tv_chambolle(img, weight=0.1, eps=0.0002, max_num_iter=200, channel_axis=-1)

        newImage = denoise_tv_bregman(img, weight=7, eps=0.001, isotropic=False)

        # Enhance edges of the image 
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
        newImage = cv2.filter2D(src=newImage, ddepth=-1, kernel=kernel)

    else:

        # The image has no noises
        newImage = img.copy()

    # Plot
    #cv2.imshow('image',newImage)
    #cv2.waitKey(0)

    return newImage


# For checking every denoising image in qsd1_w3
"""
imagesPath = 'WEEK3/qsd1_w3/'
# Get file names of the database
files = os.listdir(imagesPath)

for file in files:
        # Check if it is an image
        if file[-4:] == ".jpg":
            denoising(imagesPath + file)
"""
