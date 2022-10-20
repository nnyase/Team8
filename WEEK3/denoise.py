from crypt import methods
import cv2
import numpy as np
import glob
from utils.SimilarityMeasures import PSNR, meanPSNR, cleanImageIndex, SSIM, noise_checker
import statistics
import argparse
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Generate median,bilateral and gaussian denoise')
    parser.add_argument('-oDir','--original_dir',type=str,help='Original/non-augment directory of images')
    parser.add_argument('-nDir','--noisy_dir',type=str,help='Noisy directory of images')
    parser.add_argument('-m','--method',type=str,help='Method of denoise: median, bilateral,gaussian, all')
    parser.add_argument('-sIndex','--skip_index',type=bool,help='Skip Index for same images',default=[])
    return parser.parse_args()


def denoise_images(noisy_imgs, method="median",skip_index=[]):
    """ Denoise a list of images based on chosen method name
    Parameters
    --------------
    noisy: list of noisy images.
    method_name: Method to pick for denoising images [ "median", "bilateral",
    "gaussian", "nlmeans"]
    ----------
    returns 
    a list of denoised_images. 
    """
    denoised_imgs = []

    if method == 'median':
        for index,image in enumerate(noisy_imgs):
            if index not in skip_index:
                denoised_imgs.append(denoise_median(noisy_imgs[index]))


        #denoised_imgs = [denoise_median(noisy_imgs[noisy_img]) for noisy_img in range(len(noisy_imgs)) if noisy_img not in skip_index noisy_img[]]
    elif method == 'bilateral':
        for index,image in enumerate(noisy_imgs):
            if index not in skip_index:
                denoised_imgs.append(denoise_bilateral(noisy_imgs[index]))

    elif method == 'gaussian':
        for index,image in enumerate(noisy_imgs):
            if index not in skip_index:
                denoised_imgs.append(denoise_gaussian(noisy_imgs[index]))

    elif method == "nlmean":
        for index,image in enumerate(noisy_imgs):
            if index not in skip_index:
                denoised_imgs.append(denoise_nlmeans(noisy_imgs[index]))


    else: 
        print("Invaild Method")

    return denoised_imgs

def denoise_median(img):
    return cv2.medianBlur(img, 3)

def denoise_bilateral(img, d=7, sigmaColor=75, sigmaSpace=75):
    """
    Bilateral filter: Filter that preserves edges well, the rest is smoothed with a gaussian.
    d: Diameter of each pixel neighborhood.
    sigmaColor: Value of \sigma in the color space. The greater the value, the colors farther to each other will start to get mixed.
    sigmaSpace: Value of \sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their
    colors lie within the sigmaColor range.
    """
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def denoise_gaussian(img, k=(3,3), sigma=2):
    """
    """
    return cv2.GaussianBlur(src=img, ksize=k, sigmaX=sigma) 

def denoise_nlmeans(img,h=10,hcolor=10,ws=7,sws=21):
    return cv2.fastNlMeansDenoisingColored(src=img,h=10,hColor=hcolor,templateWindowSize=ws,searchWindowSize=sws)


if __name__ == "__main__":
    args= parse_args()
    original = args.original_dir + "*.jpg"
    noisy = args.noisy_dir +"*.jpg"
    images = [cv2.imread(file) for file in glob.glob(noisy)]
    clean_images = [cv2.imread(file) for file in glob.glob(original)]
    method=args.method
    
    if args.skip_index == True:
        clean_index = cleanImageIndex(clean_images=clean_images,noisy_images=images)
        denoised_images = denoise_images(images,method=method,skip_index=clean_index)
    else: 
        denoised_images = denoise_images(images,method=method,skip_index=[])

    
