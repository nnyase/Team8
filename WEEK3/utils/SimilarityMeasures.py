from math import log10, sqrt
import cv2
import numpy as np
import statistics
from skimage.metrics import structural_similarity
from skimage.restoration import estimate_sigma

def PSNR(original, compressed):
    """Function that calculates peak signal to noise ratio of two images
    --------------
    Parameters
    
    original: original/non_augmented image
    compressed: image with noise
    ---------------------
    returns
    psnr : peak signal to noise ratio (float)
    """
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def meanPSNR(clean_images,denoised_images):
    """Calculate mean PSRN for a list of orginal images and denoised images
    -------------------
    Parameters
    clean_images: list of original/non-augmented images
    denoised_images: list of noisy images used to compare for PSNR Score
    ------------------------
    return
    mean_psnr: Mean score for all images float"""
    psnr_list = []
    for index,image in enumerate(denoised_images):
        psnr_list.append(PSNR(clean_images[index],image))
    print(psnr_list)
    mean_psnr=statistics.fmean(psnr_list)
    return mean_psnr


def cleanImageIndex(clean_images,noisy_images):
    """Extracts a index for the non-augmented images in two image datasets
    ---------------------------
    Parameters
    clean_images: list of original/non-augmented images
    denoised_images: list of noisy images used to compare for PSNR Score
    ----------------------------------
    return 
    clean_index: list that contains indexes of non-augmented images in the noisy images ds.
    """ 
    clean_index = []
    for index in range(len(noisy_images)):
        if PSNR(clean_images[index],noisy_images[index]) == 100:
            clean_index.append(index)
    return clean_index

def SSIM(original_images,noisy_images):
    """Calculate mean SSIM for a list of orginal images and denoised images
    -------------------
    Parameters
    clean_images: list of original/non-augmented images
    denoised_images: list of noisy images used to compare for PSNR Score
    ------------------------
    return
    mean_SSIM: Mean score for all images float
    """
    ssim_list = []
    for index, image in enumerate(noisy_images):
        ssim_list.append(structural_similarity(original_images[index],image,channel_axis =2))
    return statistics.fmean(ssim_list)


def noise_checker(images,multichannel=True):
    """
    Function that gives an estimate on how much noise a list of images has using
    a robust wavelet-based estimator of the (Gaussian) noise standard deviation.
    Parameters
    -----------------
    images: list of images
    multichannel: True if the image in RGB False if the image in grayscale.

    Returns 
    -----------------
    mean_noise: mean of amount of noise in a list of images
    """
    noise_list = []
    for image in images:
        noise = estimate_sigma(image=image,average_sigmas=True,channel_axis=2)
        noise_list.append(noise)
    return statistics.fmean(noise_list)
    



    
