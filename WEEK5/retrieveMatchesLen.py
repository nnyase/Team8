import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils.mapk import mapkL
from utils.managePKLfiles import read_pkl, store_in_pkl
import os


def evaluateDiscardF1(predicted, real):
    """
    This function estimates the F1 value of the dicarded images of the database.

    Parameters
    ----------
    predicted : list
        Predicted results.
    real : list
        Ground-truth results.

    Returns
    -------
    F : float
        F score.

    """
          
    T = 0
    TP = 0
    P = 0
    for a, p in zip(real, predicted):
        if len(a)>len(p):
            
            numPaintings = len(p)
            
        elif len(p)>len(a):
            
            numPaintings = len(a)
            
        else:
            
            numPaintings = len(a)
        
        for j in range(numPaintings):
            if a[j] == -1:
                T += 1
                
                if p[j][0] == -1:
                    TP += 1
            
            if p[j][0] == -1:
                P +=1
    if T == 0:
        precision = 1
    else:
        precision = TP/T
    if P == 0:
        recall = 1
    else:
        recall = TP/P
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall / (precision + recall)
    
    return F, precision, recall

def brutForceMatcher(path1, path2, des_type): 
    """ 
        this function takes the descriptor of every feature in first set and is matched with all other features in second set 
        using some distance calculation, only taking into account those that are below our threshold value 
    Parameters
    ----------
    path1 : string
        Path of the folder where .npy descriptor files of the database are stored.
    path2 : string
        Path of the folder where .npy descriptor files of the query images are stored..
    des_type : str
        Type of the descriptors.
    Returns
    -------
    matches : list of lists (int)
        Matches found in the image.
    """
    des1 =  np.load(path1)
    des2   = np.load(path2)
    
    # create BFMatcher object
    if des_type == "orb" or des_type == "brief":
        
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    else:
        # Default L2 norm
        bf = cv.BFMatcher(crossCheck=True)
    
    good = []
    # Match descriptors.
    if des1.shape[0] == 0  and des2.shape[0] == 0:
        
        return -100
    
    if not((des1.shape[0]==0) or (des2.shape[0]==0)):
        
        matches = bf.match(des1,des2)
        good = matches
        
    return -len(good)

def brutForceMatcherKnn(path1, path2, des_type): 
    """ 
        this function takes the descriptor of every feature in first set and is matched with all other features in second set 
        using some distance calculation, only taking into account those that are below our threshold value 
    Parameters
    ----------
    path1 : string
        Path of the folder where .npy descriptor files of the database are stored.
    path2 : string
        Path of the folder where .npy descriptor files of the query images are stored.
    des_type : str
        Type of the descriptors.
    Returns
    -------
    matches : list of lists (int)
        Matches found in the image.
    """
    des1 = np.load(path1)
    des2 = np.load(path2) 
    
    # create BFMatcher object
    if des_type == "orb" or des_type == "brief":
        
        bf = cv.BFMatcher(cv.NORM_HAMMING)
    
    else:
        # Default L2 norm
        bf = cv.BFMatcher()
    
    good = []
    
    
    if des1.shape[0] == 0  and des2.shape[0] == 0:
        
        return -100
    
    if not((des1.shape[0]==0) or (des2.shape[0]==0)):
        
        matches = bf.knnMatch(des1,des2,k=2)
        if not(len(matches) > 0 and len(matches[0])==1):
            # Apply ratio test
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append([m])
                
    return -len(good)

#FLANN based Matcher
def flannMatcherKnn(path1, path2, des_type):

    """
    this function takes the descriptor of every feature in first set and is matched with all other features in second set 
        using some distance calculation, only taking into account those that are below our threshold value 

    ----------
    path1 : string
        Path of the folder where .npy descriptor files of the database are stored.
    path2 : string
        Path of the folder where .npy descriptor files of the query images are stored.
    des_type : str
        Type of the descriptors.
    Returns
    -------
    matches : list of lists (int)
        Matches found in the image.
    """

    descriptors1 = np.load(path1)
    descriptors2 = np.load(path2)

    # FLANN parameters
    if des_type == 'orb' or des_type == "brief":
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

    # Empty array => no matching so give a big vector to have a big distance
    if descriptors1.shape[0] == 0  and descriptors2.shape[0] == 0:
        
        return -100
    
    if not((descriptors1.shape[0]<2) or (descriptors2.shape[0]<2)):
        good = []
        matches = flann.knnMatch(descriptors1,descriptors2,k=2)
        
        
        
        #Ratio test as per Lowe's paper
        for i in range(len(matches)):
            if len(matches[i])==2:
                m,n = matches[i]
                if m.distance < 0.7*n.distance:
                    good.append([getattr(m,'distance')])

        return -len(good)

    else:

        return 0

def saveBestKmatchesLocalDes(bbddDescriptorsPath, qDescriptorsPath, k, matchingFunc, des_type, 
                             discardMinLen):
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
    matchingFunc : str
        Matching function name.
    des_type : str
        Type of the descriptors.
    discardMinLen: int
        Minimum number of matches not to discard.
    Returns
    -------
    result : list of lists of lists (int)
        The best k matches for each image in the query. The k matches are sorted from
        the most similar to the least one.
    """
    
    # Get matching func
    if matchingFunc == "bf":
        matchingFunc = brutForceMatcher
    elif matchingFunc == "bfknn":
        matchingFunc = brutForceMatcherKnn
    elif matchingFunc == "flannknn":
        matchingFunc = flannMatcherKnn
    
    # Compute number of images in each set
    numBBDD = len(os.listdir(bbddDescriptorsPath))
    numQ = len(os.listdir(qDescriptorsPath))

    # Init result list
    result = []

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
            distance = matchingFunc(descriptors_Q1_Path,descriptors_DDBB_Path, des_type)
            
            # Save distance
            distances[j] = distance

        # Sort the distances and get k smallest values indexes
        sortedIndexes = np.argsort(distances)
        
        # If distance too far put [-1]
        if min(distances) >-discardMinLen:
            if int(fileQ[:-4].split("_")[-1]) == 0:
                result.append([[-1]])
            else:
                result[-1].append([-1])
        else:
            if int(fileQ[:-4].split("_")[-1]) == 0:
                result.append([sortedIndexes[:k].tolist()])
            else:
                result[-1].append(sortedIndexes[:k].tolist())
    
    return result  

def bestThreshold():

    """
        This function print the best threshold value and the mapkL that we obtain with this threshold value
    """
    maxResult = 0
    bestThreshold = 0

    actual = read_pkl(pathStore2)
    for threshold in range(-100,-40,10):
        result = mapkL(actual, saveBestKmatchesLocalDes(path11, path22, 5, "flann", "sift", threshold), k=5)
        if result > maxResult:
            maxResult = result
            bestThreshold = threshold
    print("Threshold = " + str(bestThreshold) + " ==> " + str(maxResult))


path11="./descriptors/BBDD/local_descriptor/sift_300/"
path22="./descriptors/qsd1_w4/local_descriptor/sift_method4_300/"
pathStore1 = './result/result_sift_brutForceMatcher.pkl' 
pathStore2 = './result/gt_corresps.pkl'
pathStore3 = './result/result_sift_flann.pkl'

# store_in_pkl(pathStore1, saveBestKmatches(path11, path22, 10, "sift"))
# print(mapkL(read_pkl(pathStore2), read_pkl(pathStore1), k=5))

#bestThreshold()