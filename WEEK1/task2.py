import cv2
import numpy as np
import matplotlib.pyplot as plt #importing matplotlib
import os
import sys


#SimilarityFromDescriptors def:
###############################################################################
# By givin to descriptor paths to the "def SimilarityFromDescriptor" It'll    #
# calculate the similarity using  Euclidean distance, L1 distance, Hellinger  #
# kernel and (will add more today), then a graph will be displayed showing    #
# both histograms and will print the values on the console                    #
###############################################################################    
## Instance declared at line 130



#SimilarityFromImages def:
##################################################################################
# The same that  SimilarityFromDescriptors but builds the concatenated histogram #
# arrays from the images                                                         #
################################################################################## 
##Not used (Ask professor, slides says this script should be used with images)






def SimilarityFromDescriptors(path1,path2):
    DDBB = np.load(path1)
    Q1   = np.load(path2) 
    EuclidianDistance(DDBB, Q1)
    L1_Distance(DDBB, Q1)
    X2_Distance(DDBB, Q1)
    Hellinger_kernel(DDBB, Q1)
    plt.plot(DDBB)
    plt.plot(Q1)
    plt.show()
    return

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
        print("Euclidian distance")
        print(ED_Result)
    return ED_Result
  
  
#L1 distance
def L1_Distance(con,testCon):
    integral=0
    for i in range(len(con)):
        L1= abs(con[i]-testCon[i])
        integral= integral + L1
    
    else:
        L1_Result=integral
        print("L1 distance")
        print(L1_Result)
    return L1_Result
    
#X2 distance
def X2_Distance(con,testCon):
    integral=0
    for i in range(len(con)):
        x2= ((con[i]-testCon[i])**2)/(con[i]+testCon[i])
        integral= integral + x2
    
    else:
        x2_Result=integral
        print("x2 distance")
        print(x2_Result)
    return x2_Result
  
#Hellinger kernel 
def Hellinger_kernel(con,testCon):
    integral=0
    for i in range(len(con)):
        HK= ((con[i]*testCon[i])**.5)
        integral= integral + HK
    
    else:
        HK_Result=integral
        print("Hellinger Kernel distance")
        print(HK_Result)
    return HK_Result


pathfromBBDD = "./descriptorsBBDD/bbdd_00067.npy"
pathfromQ1 = "./descriptorsQ1/bbdd_00027.npy"
SimilarityFromDescriptors(pathfromBBDD,pathfromQ1)
