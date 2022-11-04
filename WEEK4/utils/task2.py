import numpy as np
import cv2 as cv
from numpy.linalg import norm
from utils.mapk import mapk
from utils.distanceMetrics import EuclidianDistance, L1_Distance, X2_Distance, Hellinger_distance, Cosine_Similarity
from managePKLfiles import read_pkl, store_in_pkl
import os
import matplotlib as plt

# L2 Norm of a Vector
def vectorNorm(descriptors1, descriptors2):
    vector = flannMatcher(descriptors1, descriptors2)
    return norm(vector)

#FLANN based Matcher
def flannMatcher(descriptors1, descriptors2, local = 'SIFT', b = 10):

    """
    This function gives best k local matches and return a vector containing distances of k best matches

    Parameters
    ----------
    descriptors1 : np array
    descriptors2 : np array
    local : string
        Type of local descriptors
    b : integer
        Size of the result vector => best b distance value
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

    # Empty array => no matching so give a big vector to have a big distance
    if not (descriptors1.size == 0 or descriptors2.size == 0):

        matches = flann.knnMatch(descriptors1,descriptors2,k=2)

        #Ratio test as per Lowe's paper
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append([getattr(m,'distance')])

        #Keep only 5 best matches
        good = sorted(good[:b])

        #Convert it into numpy array
        vector = np.array(good)

        # If the vector is empty or has few matches
        if len(good) < b:
        
            return np.full(b, 500)

        return vector

    else:

        return np.full(b, 500)

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
    result : list of lists of lists (int)
        The best k matches for each image in the query. The k matches are sorted from
        the most similar to the least one.

    """
    
    # Compute number of images in each set
    numBBDD = len(os.listdir(bbddDescriptorsPath))
    numQ = len(os.listdir(qDescriptorsPath))
    
    
    # Get distance function
    if distanceFunc == "euclidean":
        distanceFunc = EuclidianDistance
    elif distanceFunc == "l1":
        distanceFunc = L1_Distance
    elif distanceFunc == "x2":
        distanceFunc = X2_Distance
    elif distanceFunc == "hellinger":
        distanceFunc = Hellinger_distance
    elif distanceFunc == "cosSim":
        distanceFunc = Cosine_Similarity
    elif distanceFunc == "vectorNorm":
        distanceFunc = vectorNorm
        
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
            distance = SimilarityFromDescriptors(descriptors_Q1_Path,
                                                 descriptors_DDBB_Path,False, distanceFunc)

            # Save distance
            distances[j] = distance

        # Sort the distances and get k smallest values indexes
        sortedIndexes = np.argsort(distances)

        # If distance too far put [-1]
        if min(sorted(distances)) > 600:
            result.append([-1])
        else:
            if int(fileQ[:-4].split("_")[-1]) == 0:
                result.append([sortedIndexes[:k].tolist()])
            else:
                result[-1].append(sortedIndexes[:k].tolist())
    
    return result    

###############################################################################
# By givin two descriptor paths to the "def SimilarityFromDescriptor" It'll   #
# calculate the similarity using  Euclidean distance, L1 distance, Hellinger  #
# kernel and (will add more ), then a graph will be displayed showing         #
# both histograms (if the last parameter is activated) and will print the     #
#values on the console                                                        #
############################################################################### 
def SimilarityFromDescriptors(path1,path2,activatePlot , distanceFunction):
    
    # Get distance function
    if distanceFunction == "euclidean":
        distanceFunction = EuclidianDistance
    elif distanceFunction == "l1":
        distanceFunction = L1_Distance
    elif distanceFunction == "x2":
        distanceFunction = X2_Distance
    elif distanceFunction == "hellinger":
        distanceFunction = Hellinger_distance
    elif distanceFunction == "cosSim":
        distanceFunction = Cosine_Similarity
    elif distanceFunction == "vectorNorm":
        distanceFunction = vectorNorm
        
    # Load descriptors
    DDBB = np.load(path1)
    Q1   = np.load(path2) 
    
    # Calculate distance
    distance = distanceFunction(DDBB, Q1)
    
    if activatePlot:      
        plt.plot(DDBB)
        plt.plot(Q1)
        plt.show()
    
    return distance


# COMPUTE PKL FILE
bbddDescriptorsPath = 'WEEK4/descriptors/BBDD/local_descriptor/sift/' 
qDescriptorsPath = 'WEEK4/descriptors/qsd1_w4/local_descriptor/sift_method4/'
k = 5
distanceFunc = 'vectorNorm'
pathStore1 = 'WEEK4/result/result_sift_flannMatcher.pkl' 

store_in_pkl(pathStore1, saveBestKmatches(bbddDescriptorsPath, qDescriptorsPath, k, distanceFunc))

# COMPUTE MAPKL 
pathStore2 = 'WEEK4/result/gt_corresps.pkl'
print(mapk(read_pkl(pathStore1), read_pkl(pathStore2), k=5))

