import cv2
import numpy as np
import matplotlib.pyplot as plt #importing matplotlib
import os
import sys


#############################################
##Task 3 starts at line 135                 #
#############################################





#SimilarityFromDescriptors def:
###############################################################################
# By givin two descriptor paths to the "def SimilarityFromDescriptor" It'll   #
# calculate the similarity using  Euclidean distance, L1 distance, Hellinger  #
# kernel and (will add more ), then a graph will be displayed showing         #
# both histograms (if the last parameter is activated) and will print the     #
#values on the console                                                        #
###############################################################################    

#SimilarityFromImages def:
##################################################################################
# The same that  SimilarityFromDescriptors but builds the concatenated histogram #
# arrays from the images                                                         #
################################################################################## 
##Not used (Ask professor, slides says this script should be used with images)






def SimilarityFromDescriptors(path1,path2,activatePlot , distanceFunction):
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
    Hellinger_kernel(con1,con2)
    plt.plot(con1)
    plt.plot(con2)
    plt.show()
    cv2.waitKey(0)
    return


#EuclidianDistance
def EuclidianDistance(con,testCon):
    integral=0
    for i in range(len(con)):
        EuclidianDistance= ((con[i]-testCon[i])**2)
        integral= integral + EuclidianDistance
    else:
        ED_Result=integral**.5
    #    print("Euclidian distance")
    #    print(ED_Result)
    return ED_Result
  
  
#L1 distance
def L1_Distance(con,testCon):
    integral=0
    for i in range(len(con)):
        L1= abs(con[i]-testCon[i])
        integral= integral + L1
    
    else:
        L1_Result=integral
    #    print("L1 distance")
    #    print(L1_Result)
    return L1_Result
    
#X2 distance
def X2_Distance(con,testCon):
    integral=0
    for i in range(len(con)):
        x2= ((con[i]-testCon[i])**2)/(con[i]+testCon[i])
        integral= integral + x2
    
    else:
        x2_Result=integral
    #    print("x2 distance")
    #    print(x2_Result)
    return x2_Result
  
#Hellinger kernel 
def Hellinger_kernel(con,testCon):
    integral=0
    for i in range(len(con)):
        HK= ((con[i]*testCon[i])**.5)
        integral= integral + HK
    
    else:
        HK_Result=integral
    #    print("Hellinger Kernel distance")
    #    print(HK_Result)
    return HK_Result


#####TASK 3 #################################################
###### Same as Task2 but applied to all descriptors on DDBB 
###### for each descriptor in Q1
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
  

# Task4
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
    
    # Create results list of lists
    result = [[-1.]*k for i in range(28)]
    
    
    # For every image in query
    for i in range(28):
        
        # Get descriptor path
        descriptors_Q1_Path = qDescriptorsPath + PathBuilder(i)
        
        # Create list of distances
        distances = np.array([-1.]*285)
        
        # For every image in BBDD
        for j in range(285):
            
            # Get descriptor path
            descriptors_DDBB_Path = bbddDescriptorsPath + "bbdd_" + PathBuilder(j)
            
            # Calculate distance
            distance = SimilarityFromDescriptors(descriptors_Q1_Path,
                                                 descriptors_DDBB_Path,False, distanceFunc)
            
            # Save distance
            distances[j] = distance

        # Sort the distances and get k smallest values indexes
        sortedIndexes = np.argsort(distances)
        
        # Save results in the list
        result[i][:] = sortedIndexes[:k]
    
    return result

pathfromBBDD = "./descriptors/descriptors_BBDD_rgb/" 
pathfromQ1 = "./descriptors/descriptors_qsd1_w1_rgb/"



saveBestKmatches(pathfromBBDD, pathfromQ1, 5, EuclidianDistance)










