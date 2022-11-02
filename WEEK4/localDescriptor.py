import cv2
import numpy as np

def SIFT(image):
    
    features = cv2.SIFT_create(nfeatures = 300)
    kp, des = features.detectAndCompute(image, None)
    
    if des is None:
        des = np.zeros([0,128], dtype = np.uint8)
    
    return des

def SURF(image):
    # Speeded version of SIFT = 3x faster that SIFT
    # Here I set Hessian Threshold to 400
    features = cv2.xfeatures2d.SURF_create(400)
    # Find keypoints and descriptors directly
    kp, des = features.detectAndCompute(image,None)
    
    if des is None:
        des = np.zeros([0,128], dtype = np.uint8)
    
    return des
    
def ORB(image):
    
    features = cv2.ORB_create()
    kp, des = features.detectAndCompute(image, None)
    
    if des is None:
        des = np.zeros([0,32], dtype = np.uint8)
    
    return des
    
def BRIEF(image):
    
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp = star.detect(image)
    kp, des = brief.compute(image, kp)
    
    if des is None:
        des = np.zeros([0,32], dtype = np.uint8)
    
    
    return des

def harrisLaplace(image):
    
    features = cv2.xfeatures2d.HarrisLaplaceFeatureDetector.create(maxCorners=300)
    featuresSIFT = cv2.SIFT_create(nfeatures = 300)
    
    kp = features.detect(image)
    kp, des = featuresSIFT.compute(image, kp)
    
    if des is None:
        des = np.zeros([0,128], dtype = np.uint8)
    
    return des


descriptor_types = {'sift': SIFT,
                   'surf':SURF,
                   'orb': ORB,
                   'brief': BRIEF,
                   'harrisLaplace': harrisLaplace}

def generateLocalDescriptors(descriptorType, image):
    """
    This function generates the descriptors of the image generated using the method given.

    Parameters
    ----------
    descriptorType : str
        Method to generate the descriptors.
    image : numpy array (uint8)
        Image to compute descriptors.

    Returns
    -------
    des : numpy array
        List of the descriptors of the detected keypoints.

    """
    
    des = descriptor_types[descriptorType](image)
    
    return des

# img = cv2.imread('WEEK4/denoisedImages/optimized/qsd1_w4/00000.jpg')
# SURF(img)