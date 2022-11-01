import cv2



def SIFT(image):
    
    features = cv2.SIFT_create(nfeatures = 3000)
    kp, des = features.detectAndCompute(image, None)
    
    return des
    
def ORB(image):
    
    features = cv2.ORB_create()
    kp, des = features.detectAndCompute(image, None)
    
    return des
    
def BRIEF(image):
    
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp = star.detect(image)
    kp, des = brief.compute(image, kp)
    
    return des

def harrisLaplace(image):
    
    features = cv2.xfeatures2d.HarrisLaplaceFeatureDetector.create()
    featuresSIFT = cv2.SIFT_create()
    
    kp = features.detect(image)
    kp, des = featuresSIFT.compute(image, kp)
    
    return des


descriptor_types = {'sift': SIFT, 
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


