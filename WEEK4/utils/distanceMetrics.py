from numpy import dot
import numpy as np
from numpy.linalg import norm


#EuclidianDistance
def EuclidianDistance(hist1,hist2):
    """
    -> Return the Euclidian distance between two  histogram
    """
    return (np.sum((hist1 - hist2)**2))**0.5

#L1 distance
def L1_Distance(hist1,hist2):
    """
    -> Return the L1 distance between two  histogram
    """
    return (np.sum(abs(hist1 - hist2)))

#X2 distance
def X2_Distance(hist1,hist2):
    """
    -> Return the X2 distance between two  histogram
    """
    return np.sum(np.divide((hist1 - hist2)**2, hist1 + hist2, out=np.zeros_like(hist1), where= (hist1 + hist2) !=0))

#Hellinger kernel 
def Hellinger_distance(hist1,hist2):
    """
    -> Return the Hellinger distance between two  histogram
    """
    return np.sqrt(np.sum((np.sqrt(hist1) - np.sqrt(hist2))**2))/(np.sqrt(2))
  

# Histogram Intersection
#def Hist_Intersection(con,testCon):
#    HI=0
#    for i in range(len(con)):
#        HI += min(con[i], testCon[i])
#        #print("Histogram Intersection")
#        #print(HI)
#    return -HI

# Cosine Similarity
def Cosine_Similarity(con,testCon):
    cos_sim = dot(con, testCon)/(norm(con)*norm(testCon))
    #print("Cosine similarity")
    #print(cos_sim)
    return -cos_sim