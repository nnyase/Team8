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
##Not used (Ask professor, slides says this script should be used with images for task 2)






def SimilarityFromDescriptors(path1,path2,activatePlot):
    r1 = [0, 0, 0, 0, 0] 
    results_array= np.array(r1)
    DDBB = np.load(path1)
    Q1   = np.load(path2) 
    results_array[0] = EuclidianDistance(DDBB, Q1)
    results_array[1] =L1_Distance(DDBB, Q1)
    results_array[2] =X2_Distance(DDBB, Q1)
    results_array[3] =Hellinger_kernel(DDBB, Q1)
   
    if activatePlot:      
        plt.plot(DDBB)
        plt.plot(Q1)
        plt.show()
    return results_array

def SimilarityFromImages(img_path1,img_path2,activatePlot):
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
    
    results_array = np.array([3]*1)
    results_array[0] = EuclidianDistance(con1, con2)
    results_array[1] = L1_Distance(con1,con2)
    results_array[2] = X2_Distance(con1, con2)
    results_array[3] = Hellinger_kernel(con1,con2)
    
    if activatePlot:      
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
        #print("Euclidian distance")
       # print(ED_Result)
    return ED_Result
  
  
#L1 distance
def L1_Distance(con,testCon):
    integral=0
    for i in range(len(con)):
        L1= abs(con[i]-testCon[i])
        integral= integral + L1
    
    else:
        L1_Result=integral
        #print("L1 distance")
        #print(L1_Result)
    return L1_Result
    
#X2 distance
def X2_Distance(con,testCon):
    integral=0
    for i in range(len(con)):
        x2= ((con[i]-testCon[i])**2)/(con[i]+testCon[i])
        integral= integral + x2
    
    else:
        x2_Result=integral
        #print("x2 distance")
        #print(x2_Result)
    return x2_Result
  
#Hellinger kernel 
def Hellinger_kernel(con,testCon):
    integral=0
    for i in range(len(con)):
        HK= ((con[i]*testCon[i])**.5)
        integral= integral + HK
    
    else:
        HK_Result=integral
        #print("Hellinger Kernel distance")
        #print(HK_Result)
    return HK_Result


#####TASK 3#################################################
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
  
pathfromBBDD = "./descriptors/descriptors_BBDD_rgb/bbdd_" 
pathfromQ1 = "./descriptors/descriptors_qsd1_w1_rgb/"
pathResultFiles= "./task_4_Results/"




#####TASK 4#################################################
##evaluate each Q1 data against 285 descriptors
##Save the data in a 2d vector
##Evaluate and return the top K 
Q1_Resutls_array = np.zeros((17,285))

########
for i in range(28):
    descriptors_Q1_Path = pathfromQ1 + PathBuilder(i)
    for j in range(285):
        descriptors_DDBB_Path = pathfromBBDD + PathBuilder(j)
       # Q1_Resutls_array[i,j]  =  SimilarityFromDescriptors(descriptors_Q1_Path,descriptors_DDBB_Path,False)
        print ("Q1 No:")
        print(str(i))
        print ("vs ddbb No:")
        print(str(j))
        
        
        
    










