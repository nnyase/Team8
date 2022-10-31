import cv2



def getKeypoints(keyPointType, image):
    """
    This function generates the keypoints of the image generated using the method given.

    Parameters
    ----------
    keyPointType : str
        Method to generate the keypoints.
    image : numpy array (uint8)
        Image to compute keypoints.

    Returns
    -------
    keyPoints : list
        List of detected keypoints.

    """
    
    # Get keypoint generation algorithm
    if keyPointType == "harris_laplace":
        
        features = cv2.xfeatures2d.HarrisLaplaceFeatureDetector.create()
        
    elif keyPointType == "sift":
        
        features = cv2.SIFT_create()
        
    elif keyPointType == "surf":
        
        features = cv2.xfeatures2d.SURF_create(400)
        
    elif keyPointType == "star":
        
        features = cv2.xfeatures2d.StarDetector_create()

    elif keyPointType == "orb":
        
        features = cv2.ORB_create()

    # Get keypoints
    keyPoints = features.detect(image)
    
    return keyPoints


def getDescriptors(descriptorType, keyPoints, image):
    """
    This function generates the descriptors of the image generated using the keypoints and method given.

    Parameters
    ----------
    descriptorType : str
        Method to generate the descriptors.
    keyPoints : list
        List of the keypoints of the image.
    image : numpy array (uint8)
        Image to compute keypoints.

    Returns
    -------
    des : list
        List of generated descriptors.

    """
    
    # Get descriptor generation algorithm
    if descriptorType == "sift":
        
        features = cv2.SIFT_create()
        
    elif descriptorType == "surf":
        
        features = cv2.xfeatures2d.SURF_create(400)
        
    elif descriptorType == "brief":
        
        features = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    elif descriptorType == "orb":
        
        features = cv2.ORB_create()

    # Get descriptors
    keyPoints, des = features.compute(image, keyPoints)
    
    return des