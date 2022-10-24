from denoise import denoise_images
from utils.SimilarityMeasures import SSIM
import os
import cv2
import numpy as np

def computeMeanSSIM(noisyImagesPath, originalImagesPath):
    
    noisy_images = [cv2.imread(noisyImagesPath + file) for file in os.listdir(noisyImagesPath) if file[-4:]==".jpg"]
    
    denoised_images = denoise_images(noisy_images, "optimized")
    
    original_images = [cv2.imread(originalImagesPath + file) for file in os.listdir(originalImagesPath) if file[-4:]==".jpg"]
    
    sim = SSIM(original_images, denoised_images)
    
    print(sim)
    

noisyImagesPath = "../../WEEK3/qsd2_w3/"
originalImagesPath = "../../WEEK3/qsd2_w3/non_augmented/"

computeMeanSSIM(noisyImagesPath,originalImagesPath)