import numpy as np
import os
from utils.mapk import mapkL
from utils.managePKLfiles import read_pkl
from utils.computeRetrieval import SimilarityFromDescriptors, SimilarityFromText

# New savesBestKmatches
def saveBestKmatchesNew(bbddDescriptorsPathText = None, bbddDescriptorsPathColor = None, bbddDescriptorsPathTexture = None, 
                        qDescriptorsPathText = None, qDescriptorsPathColor = None, qDescriptorsPathTexture = None, k = 10, 
                        distanceFuncText = None, distanceFuncColor = None, distanceFuncTexture = None, weightText = 1, weightColor = 1, 
                        weightTexture = 1):
    """ This function computes all the similarities between the database and query images
        using distances functions given and returns k best matches for every query image
    

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

    # Get names of files
    if bbddDescriptorsPathText != None:
        bbddDescriptorsPath = bbddDescriptorsPathText
        qDescriptorsPath = qDescriptorsPathText

    elif bbddDescriptorsPathColor != None:
        bbddDescriptorsPath = bbddDescriptorsPathColor
        qDescriptorsPath = qDescriptorsPathColor
    else:
        bbddDescriptorsPath = bbddDescriptorsPathTexture
        qDescriptorsPath = qDescriptorsPathTexture
    
    # Compute number of images in each set
    numBBDD = len(os.listdir(bbddDescriptorsPath))
        
    # Init result list
    result = []
    
    # For every image in query
    for i, fileQ in enumerate(sorted(os.listdir(qDescriptorsPath))):
        
        # Create list of distances
        distances = np.array([-1.]*numBBDD)
        
        # For every image in BBDD
        for j, fileBBDD in enumerate(sorted(os.listdir(bbddDescriptorsPath))):

            # Mean weighted distance
            distance = 0

            # Calculate distance
            if qDescriptorsPathText != None:
                distanceText = SimilarityFromText(qDescriptorsPathText + fileQ,
                                                bbddDescriptorsPathText + fileBBDD,False, distanceFuncText)
                # Distance for text descriptors is in [0,1]
                distance += weightText*distanceText

            if qDescriptorsPathColor != None:
                distanceColor = SimilarityFromDescriptors(qDescriptorsPathColor + fileQ,
                                                bbddDescriptorsPathColor + fileBBDD,False, distanceFuncColor)
                # Distance for color descriptors is in [0,2]
                distance += weightColor*(distanceColor/2)
            if qDescriptorsPathTexture != None:
                distanceTexture = SimilarityFromDescriptors(qDescriptorsPathTexture + fileQ,
                                                bbddDescriptorsPathTexture + fileBBDD,False, distanceFuncTexture)
                # Distance for texture descriptors is in [0,2]
                distance += weightTexture*(distanceTexture/2)

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
    

def bestCoefficient(bbddDescriptorsPathText = None, bbddDescriptorsPathColor = None, bbddDescriptorsPathTexture = None, 
                    qDescriptorsPathText = None, qDescriptorsPathColor = None, qDescriptorsPathTexture = None, k = 10, 
                    distanceFuncText = 1, distanceFuncColor = "l1", distanceFuncTexture = "l1"):

    """This function return the best weight for each descriptors (text, color, texture) in order to get the better mapkL value

    Parameters
    ----------
    bbddDescriptorsPathText : string
        Path of the folder where .npy text descriptor  files of the database are stored.
    bbddDescriptorsPathColor : string
        Path of the folder where .npy color descriptor  files of the database are stored.  
    bbddDescriptorsPathTexture : string
        Path of the folder where .npy texture descriptor  files of the database are stored.  
    qDescriptorsPathText : string
        Path of the folder where .npy text descriptor files of the query images are stored.
    qDescriptorsPathColor : string
        Path of the folder where .npy color descriptor files of the query images are stored.
    qDescriptorsPathTexture : string
        Path of the folder where .npy texture descriptor files of the query images are stored..
    k : int
        Quantity of best matches is returned.
    distanceFuncText : function
        Distance function that will be used to compute the similarities of the text.
    distanceFuncColor : function
        Distance function that will be used to compute the similarities of the color.
    distanceFuncTexture : function
        Distance function that will be used to compute the similarities of the texture.

    Returns
    -------
    result : weightText, weightColor and weightTexture and bestmapkL

    """

    gtResults = read_pkl('./data/qsd1_w3/gt_corresps.pkl')
    best = 0
    besti = 0
    bestj = 0
    bestl = 0

    #2 descriptors
    if(bbddDescriptorsPathText != None and bbddDescriptorsPathColor != None and bbddDescriptorsPathTexture == None 
    or bbddDescriptorsPathText != None and bbddDescriptorsPathColor == None and bbddDescriptorsPathTexture != None 
    or bbddDescriptorsPathText == None and bbddDescriptorsPathColor != None and bbddDescriptorsPathTexture != None):

        #Text and Color
        if(bbddDescriptorsPathText != None and bbddDescriptorsPathColor != None):
            for i in [1/4,1/3,1,3,4]:

                    predictedResults = saveBestKmatchesNew(bbddDescriptorsPathText = bbddDescriptorsPathText, bbddDescriptorsPathColor = bbddDescriptorsPathColor,
                    qDescriptorsPathText = qDescriptorsPathText, qDescriptorsPathColor = qDescriptorsPathColor, distanceFuncText = distanceFuncText, distanceFuncColor = distanceFuncColor, k = k, weightText = i)

                    mapkl = mapkL(gtResults, predictedResults, 10)

                    print(round(mapkl,2))

                    if(best < mapkl):
                        besti = i
                        best = mapkl

            print("Best combination for color and text is : weightText = " + str(besti) + " and weightColor = 1 =====> " + str(best))

        #Text and Texture
        if(bbddDescriptorsPathText != None and bbddDescriptorsPathTexture != None):
            for i in [1/4,1/3,1,3,4]:
                
                    predictedResults = saveBestKmatchesNew(bbddDescriptorsPathText = bbddDescriptorsPathText, bbddDescriptorsPathTexture = bbddDescriptorsPathTexture,
                    qDescriptorsPathText = qDescriptorsPathText, qDescriptorsPathTexture = qDescriptorsPathTexture, distanceFuncText = distanceFuncText, distanceFuncTexture = distanceFuncTexture, k = k, weightText = i)

                    mapkl = mapkL(gtResults, predictedResults, 10)

                    print(round(mapkl,2))

                    if(best < mapkl):
                        besti = i
                        best = mapkl

            print("Best combination for text and texture is : weightText = " + str(besti) + " and weightTexture = 1 =====> " + str(best))

        #Text and Texture
        if(bbddDescriptorsPathColor != None and bbddDescriptorsPathTexture != None):
            for i in [1/4,1/3,1,3,4]:
                
                    predictedResults = saveBestKmatchesNew(bbddDescriptorsPathColor = bbddDescriptorsPathColor, bbddDescriptorsPathTexture = bbddDescriptorsPathTexture,
                    qDescriptorsPathColor = qDescriptorsPathColor, qDescriptorsPathTexture = qDescriptorsPathTexture, distanceFuncColor = distanceFuncColor, distanceFuncTexture = distanceFuncTexture, k = k, weightColor = i)

                    mapkl = mapkL(gtResults, predictedResults, 10)

                    print(round(mapkl,2))

                    if(best < mapkl):
                        besti = i
                        best = mapkl

            print("Best combination for color and texture is : weightColor = " + str(besti) + " and weightTexture = 1 =====> " + str(best))


    #3 descriptors

    if(bbddDescriptorsPathText != None and bbddDescriptorsPathColor != None and bbddDescriptorsPathTexture != None):

            for i in [1/5,1/4,1/2,3/4]:
                for j in [1/5,1/4,1/2,3/4]:
                    for l in [1/5,1/4,1/2,3/4]:
                
                        predictedResults = saveBestKmatchesNew(bbddDescriptorsPathColor = bbddDescriptorsPathColor, bbddDescriptorsPathTexture = bbddDescriptorsPathTexture,
                        bbddDescriptorsPathText = bbddDescriptorsPathText, qDescriptorsPathColor = qDescriptorsPathColor,
                        qDescriptorsPathTexture = qDescriptorsPathTexture, qDescriptorsPathText= qDescriptorsPathText, distanceFuncColor = distanceFuncColor, 
                        distanceFuncTexture = distanceFuncTexture, distanceFuncText= distanceFuncText, k = k, weightColor = i, weightTexture= j, weightText= l)

                        mapkl = mapkL(gtResults, predictedResults, 10)

                        print('(' + str(i) + ',' + str(j) + ',' + str(l) + ') => ' + str(round(mapkl,2)))

                        if(best < mapkl):
                            besti = i
                            bestj = j
                            bestl = l
                            best = mapkl

            print("Best combination for color, texture and text is : weightColor = " + str(besti) + ", weightTexture = " + str(bestj) + " and weightText = " + str(bestl) + " =====> " + str(best))

       


# Hog descriptors
# BBDDPathTexture = './descriptors/BBDD/hog/levels_3/features_160/'
# QPathTexture = './descriptors/qsd1_w3/hog/levels_3/features_160/'

# # Text descriptors
# BBDDPathText = './textDescriptors/BBDD_pny/'
# QPathText = './textDescriptors/denoisedImages/nlmean/qsd1_w3_npy/'

# # Color descriptors
# BBDDPathColor = './descriptors/BBDD/cielab/level_3/2D_bins_20/'
# QPathColor = './descriptors/qsd1_w3/cielab/level_3/2D_bins_20/'


# result = saveBestKmatchesNew(bbddDescriptorsPathTexture = BBDDPathTexture,
#     qDescriptorsPathTexture = QPathTexture, distanceFuncTexture= L1_Distance)

# gtResults = read_pkl('./data/qsd1_w3/gt_corresps.pkl')
# print(mapkL(gtResults, result, 10))

# bestCoefficient(bbddDescriptorsPathColor=BBDDPathColor, bbddDescriptorsPathText=BBDDPathText,
# bbddDescriptorsPathTexture= BBDDPathTexture, qDescriptorsPathColor=QPathColor,
# qDescriptorsPathText=QPathText, qDescriptorsPathTexture= QPathTexture)

