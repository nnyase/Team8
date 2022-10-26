import cv2
import numpy as np
import matplotlib.pyplot as plt #importing matplotlib
import os
from mapk import mapkL
from utils.managePKLfiles import read_pkl
from utils.distanceTextMetrics import getDistance2Strings
from utils.distanceMetrics import EuclidianDistance, L1_Distance, X2_Distance, Hellinger_distance, Cosine_Similarity




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

###############################################################################
# By givin two texts contained in npy file paths to the def SimilarityFromText#
# It'll calculate the similarity using hamming, mlipns, levenshtein, etc.     #
# then a graph will be displayed showing                                      #
# both histograms (if the last parameter is activated) and will print the     #
# values on the console                                                        #
############################################################################### 

def SimilarityFromText(path1,path2,activatePlot , distanceFunction):
    # Load descriptors
    DDBB = np.load(path1)
    Q1   = np.load(path2) 
    
    # DistanceFunction is an integer
    
    # Calculate distance
    distance = getDistance2Strings(str(DDBB), str(Q1), distanceFunction)
    
    if activatePlot:      
        plt.plot(DDBB)
        plt.plot(Q1)
        plt.show()
    
    return distance


##################################################################################
# The same that  SimilarityFromDescriptors but builds the concatenated histogram #
# arrays from the images                                                         #
################################################################################## 
def SimilarityFromImages(img_path1,img_path2):
    #Image1
    img_src=cv2.imread(img_path1)
    #Image2
    test_src=cv2.imread(img_path2)
    # resize image
    dsize = (500, 500)
    img = cv2.resize(img_src, dsize)
    test = cv2.resize(test_src, dsize)
    #RGB Histograms Image1
    histB1 = cv2.calcHist([img],[0],None,[256],[0,256])
    histG1 = cv2.calcHist([img],[1],None,[256],[0,256])
    histR1 = cv2.calcHist([img],[2],None,[256],[0,256])
    # concatenate RGB histograms in one array
    con1 = np.concatenate((histB1, histG1,histR1))
    
    #RGB Histograms Image2
    histB2 = cv2.calcHist([test],[0],None,[256],[0,256])
    histG2 = cv2.calcHist([test],[1],None,[256],[0,256])
    histR2 = cv2.calcHist([test],[2],None,[256],[0,256])
    con2 = np.concatenate((histB2,histG2,histR2))
    EuclidianDistance(con1, con2)
    L1_Distance(con1,con2)
    X2_Distance(con1, con2)
    Hellinger_distance(con1,con2)
    plt.plot(con1)
    plt.plot(con2)
    plt.show()
    cv2.waitKey(0)
    return




# Get file names
def PathBuilder(i):
    if i==0 :
       path = "00000.npy"
    elif i>0 and i<10:
        path = "0000"+str(i)+".npy"
    elif i>9 and i<100:
        path = "000"+str(i)+".npy"  
    elif  i>99:
        path = "00"+str(i)+".npy"  
    return path
  

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
        
        # Save results in the list
        if int(fileQ[:-4].split("_")[-1]) == 0:
            result.append([sortedIndexes[:k].tolist()])
        else:
            result[-1].append(sortedIndexes[:k].tolist())
    
    return result
    

def saveBestKmatchesOld(bbddDescriptorsPath, qDescriptorsPath, k, distanceFunc):
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
        
        # Save results in the list
        result[i][:] = sortedIndexes[:k].tolist()
    
    return result




# BBDDPathColor = './descriptors/BBDD/cielab/level_3/2D_bins_20/'
# QPathColor = './descriptors/qsd1_w3/cielab/level_3/2D_bins_20/'

# result = saveBestKmatches(BBDDPathColor, QPathColor, 10, 'l1')
# gtResults = read_pkl('./data/qsd1_w3/gt_corresps.pkl')
# print(mapkL(gtResults, result, 10))




















