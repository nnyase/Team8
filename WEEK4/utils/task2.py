import numpy as np
import cv2 as cv

#FLANN based Matcher
def flannMatcher(descriptors1, descriptors2, local = 'SIFT'):

    """
    This function gives best k local matches and return a vector containing distances of k best matches

    Parameters
    ----------
    descriptors1 : np array
    descriptors2 : np array
    local : string
        Type of local descriptors
        
    Returns
    -------
    result : list of k best distance 
    
    """

    # FLANN parameters
    if local == 'ORB':
        #For ORB
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    else:
        #For SIFT, SURF, etc.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descriptors1,descriptors2,k=2)

    #Ratio test as per Lowe's paper
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append([getattr(m,'distance')])

    #Keep only 5 best matches
    good = sorted(good[:5])

    #Convert it into numpy array
    vector = np.array(good)

    #Print distance 
    # for match in good:
    #     print(str(getattr(match[0],'trainIdx')) + ";" + str(getattr(match[0],'queryIdx')) + "=>" + str(getattr(match[0],'distance')))

    return vector



