import numpy as np
import os
from backgroundRemoval import generateBackgroundMasks
from computeRetrieval import SimilarityFromDescriptors
from utils.distanceMetrics import EuclidianDistance, L1_Distance, X2_Distance, Hellinger_distance, Cosine_Similarity

# New savesBestKmatches

def saveBestKmatchesNew(bbddDescriptorsPathText = None, bbddDescriptorsPathColor = None, bbddDescriptorsPathTexture = None, qDescriptorsPathText = None, qDescriptorsPathColor = None, qDescriptorsPathTexture = None, k = 10, distanceFuncText = None, distanceFuncColor = None, distanceFuncTexture = None, weightText = 1, weightColor = 1, weightTexture = 1):
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
    distanceFuncText : function
        Distance function that will be used to compute the similarities of the text.
    distanceFuncColor : function
        Distance function that will be used to compute the similarities of the color.
    distanceFuncTexture : function
        Distance function that will be used to compute the similarities of the texture.
    weightText : float
        This float is in [0,1] and gives the importance of text compared to color and texture to find best matches
    weightColor : float
        This float is in [0,1] and gives the importance of color compared to text and texture to find best matches
    weightTexture : float
        This float is in [0,1] and gives the importance of texture compared to color and text to find best matches

    Returns
    -------
    result : list of lists of lists (int)
        The best k matches for each image in the query. The k matches are sorted from
        the most similar to the least one.

    """
    
    # Compute number of images in each set
    numBBDD = len(os.listdir(bbddDescriptorsPathText))
    numQ = len(os.listdir(qDescriptorsPathText))
        
    # Init result list
    result = []
    
    # For every image in query
    for i, fileQ in enumerate(os.listdir(qDescriptorsPathText)):
        
        # Create list of distances
        distances = np.array([-1.]*numBBDD)
        
        # For every image in BBDD
        for j, fileBBDD in enumerate(os.listdir(bbddDescriptorsPathText)):

            # Mean weighted distance
            distance = 0
            
            # Calculate distance
            if distanceFuncText != None:
                distanceText = SimilarityFromDescriptors(qDescriptorsPathText + fileQ,
                                                bbddDescriptorsPathText + fileBBDD,False, distanceFuncText)
                distance += weightText*distanceText
            if distanceFuncColor != None:
                distanceColor = SimilarityFromDescriptors(qDescriptorsPathColor + fileQ,
                                                bbddDescriptorsPathColor + fileBBDD,False, distanceFuncColor)
                distance += weightColor*distanceColor
            if distanceFuncTexture != None:
                distanceTexture = SimilarityFromDescriptors(qDescriptorsPathTexture + fileQ,
                                                bbddDescriptorsPathTexture + fileBBDD,False, distanceFuncTexture)
                distance += weightTexture*distanceTexture
            
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
    


