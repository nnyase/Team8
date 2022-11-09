import cv2
import glob 
import argparse 
from skimage.restoration import estimate_sigma  
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Generate median,bilateral and gaussian denoise')
    parser.add_argument('-nDir','--noisy_dir',type=str,help='Original Imgaes to denoise')
    parser.add_argument('-m','--method',type=str,help='Method of denoise: median, bilateral,gaussian, (Best) optimized')
    return parser.parse_args()


def denoise_images(noisy_imgs, method="median"):
    """ Denoise a list of images based on chosen method name
    Parameters
    --------------
    noisy: list of noisy images.
    method_name: Method to pick for denoising images [ "median", "bilateral",
    "gaussian", "nlmeans","optimized"]
    ----------
    returns 
    a list of denoised_images. 
    """
    denoised_imgs = []

    if method == 'median':
        for index,image in enumerate(noisy_imgs):
            denoised_imgs.append(denoise_median(noisy_imgs[index]))

        #denoised_imgs = [denoise_median(noisy_imgs[noisy_img]) for noisy_img in range(len(noisy_imgs)) if noisy_img not in skip_index noisy_img[]]
    elif method == 'bilateral':
        for index,image in enumerate(noisy_imgs):
            denoised_imgs.append(denoise_bilateral(noisy_imgs[index]))

    elif method == 'gaussian':
        for index,image in enumerate(noisy_imgs):
            denoised_imgs.append(denoise_gaussian(noisy_imgs[index]))

    elif method == "nlmean":
        for index,image in enumerate(noisy_imgs):
            denoised_imgs.append(denoise_nlmeans(noisy_imgs[index]))

    elif method == "optimized":
        for index,image in enumerate(noisy_imgs):
            #print(index)
            denoised_imgs.append(optimizedDenoising(noisy_imgs[index]))

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


def optimizedDenoising(img):

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

    #print(estimate_sigma(img, average_sigmas=True, channel_axis = 2))
    if estimate_sigma(img, average_sigmas=True, channel_axis = 2) >= 5: #2
        

        return cv2.medianBlur(img, 3)

    else:

        return img.copy()


def denoiseImages(queryDir, outputDir, method):
    """
    This functions denoises and stores in the output path the images in the given input path using the selected method

    Parameters
    ----------
    queryDir : str
        Input path where noisy images are.
    outputDir : str
        Output path where denoised images will be saved.
    method : str
        Method to use to denoise images.

    Returns
    -------
    None.

    """
    # Read images
    noisyImagesPath = queryDir +"/*.jpg"
    noisyImages = [cv2.imread(file) for file in glob.glob(noisyImagesPath)]
    
    # Denoise images
    denoisedImages = denoise_images(noisyImages,method=method)
    
    # Save files
    fileNames = os.listdir(queryDir)
    jpgFiles =[]
    for name in fileNames:
        if name[-4:] == ".jpg":
            jpgFiles.append(name)
    
    for index in range(len(denoisedImages)):
        cv2.imwrite(outputDir + jpgFiles[index], denoisedImages[index])
        


if __name__ == "__main__":
    args= parse_args()
    noisy = args.noisy_dir +"/*.jpg"
    images = [cv2.imread(file) for file in glob.glob(noisy)]
    method=args.method
    denoised_images = denoise_images(images,method=method)
    fileNames = os.listdir(args.noisy_dir)
    jpgFiles =[]
    for name in fileNames:
        if name[-4:] == ".jpg":
            jpgFiles.append(name)

    path = './denoisedImages/' + method + "/"
    if "qsd1_w3" in noisy:
        path_d1 = path + "qsd1_w3/"
        if not os.path.exists(path_d1):
            os.makedirs(path_d1)
        for index,image in enumerate(denoised_images):
            
            cv2.imwrite(path_d1 + jpgFiles[index], denoised_images[index])
        

    elif "qsd2_w3" in noisy:
        path_d2 = path + "qsd2_w3/"
        if not os.path.exists(path_d2):
            os.makedirs(path_d2)
        for index,image in enumerate(denoised_images):
            cv2.imwrite(path_d2 + jpgFiles[index], denoised_images[index])

