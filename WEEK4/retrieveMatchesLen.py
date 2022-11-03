import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
def brutForceMatcher(path1,path2,function,thresholdVal): 
    """ 
        this function takes the descriptor of every feature in first set and is matched with all other features in second set 
        using some distance calculation, only taking into account those that are below our threshold value 
    Parameters
    ----------
    path1 : string
        Path of the folder where .npy descriptor files of the database are stored.
    path2 : string
        Path of the folder where .npy descriptor files of the query images are stored..
    function : function
        Distance function that will be used to compute the similarities.
    Returns
    -------
    matches : list of lists (int)
        Matches found in the image.
    """
    des1 =  np.load(path1,allow_pickle=True)
    des2   = np.load(path2,allow_pickle=True)
    # create BFMatcher object
    if function ==0:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    if function ==1:
        bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)
    if function ==3:
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    if function ==4:
        bf = cv.BFMatcher(cv.NORM_, crossCheck=True)
    if function > 4:
        print ("Default selected: HAMMING")
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
   
    # Match descriptors.
    matches = bf.match(des1,des2)
    good = []
    for m in matches:
        if m.distance < thresholdVal:
            good.append([m])
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    #Return the numbers of the matches found  
    return len(good)

def brutForceMatcherKnn(path1,path2,thresholdVal): 
    """ 
        this function takes the descriptor of every feature in first set and is matched with all other features in second set 
        using some distance calculation, only taking into account those that are below our threshold value 
    Parameters
    ----------
    path1 : string
        Path of the folder where .npy descriptor files of the database are stored.
    path2 : string
        Path of the folder where .npy descriptor files of the query images are stored..
    Returns
    -------
    matches : list of lists (int)
        Matches found in the image.
    """
    des1 = np.load(path1,allow_pickle=True)
    des2 = np.load(path2,allow_pickle=True) 
   
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < thresholdVal*n.distance:
            good.append([m])
    return len(good)
def saveBestKmatches(bbddDescriptorsPath, qDescriptorsPath, k, distanceFunc):
    """ This function computes all the similarities between the database and query images
        using the distance function given and returns k best matches for every query image
    
    Parameters
    ----------
    bbddDescriptorsPath : string
        Path of the folder where .npy descriptor files of the database are stored.
    qDescriptorsPath : string
        Path of the folder where .npy descriptor files of the query images are stored..
    k : int
        Quantity of best matches is returned.
    distanceFunc : function
        Distance function that will be used to compute the similarities.
    Returns
    -------
    result : list of lists (int)
        The best k matches for each image in the query. The k matches are sorted from
        the most similar to the least one.
    """
    
    # Compute number of images in each set
    numBBDD = len(os.listdir(bbddDescriptorsPath))
    numQ = len(os.listdir(qDescriptorsPath))
    
    # Create results list of lists
    result = [[-1.]*k for i in range(numQ)]
    
    path1="./descriptors/BBDD/local_descriptor/orb/bbdd_00286_0.npy"
    path2="./descriptors/qsd1_w4/local_descriptor/orb_method3/00017_0.npy"


    # For every image in query
    for i, fileQ in enumerate(sorted(os.listdir(qDescriptorsPath))):
        
        # Get descriptor path
        descriptors_Q1_Path = qDescriptorsPath + fileQ
        
        # Create list of distances
        distances = np.array([-1.]*numBBDD)
        
        # For every image in BBDD
        for j, fileBBDD in enumerate(sorted(os.listdir(bbddDescriptorsPath))):
            

            # Get descriptor path
            descriptors_DDBB_Path = bbddDescriptorsPath + fileBBDD
            
            # Calculate distance
           
            # #distance = matchesFound(descriptors_Q1_Path, descriptors_DDBB_Path)
            # # distance = matchesFound(path1, path2)
            #distance= brutForceMatcherKnn(descriptors_Q1_Path,descriptors_DDBB_Path,0.5)
            distance=brutForceMatcher(descriptors_Q1_Path,descriptors_DDBB_Path,1,10)
            
            
            
            # Save distance
            distances[j] = distance

        # Sort the distances and get k smallest values indexes
        sortedIndexes = np.argsort(distances)
        
        # Save results in the list
        result[i][:] = sortedIndexes[:k].tolist()
    
    return result




path1="./descriptors/BBDD/local_descriptor/orb/"
path2="./descriptors/qsd1_w4/local_descriptor/orb_method3/"
path11="./descriptors/BBDD/local_descriptor/orb/"
path22="./descriptors/qsd1_w4/local_descriptor/orb_method3/"

h=saveBestKmatches(path1, path2, 5, 5)
#brutForceMatcher(path1,path2,1,5)
# h=brutForceMatcherKnn(path1,path2,0.5) 
# print(h)