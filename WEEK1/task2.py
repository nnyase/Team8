import cv2
import numpy as np
import matplotlib.pyplot as plt #importing matplotlib
import os
import sys

#Image from Query
img_src=cv2.imread("./BBDD/bbdd_00067.jpg")
#Image from datbase
test_src=cv2.imread("./qsd1_w1/00027.jpg")
# dsize
dsize = (500, 500)

# resize image
img = cv2.resize(img_src, dsize)
test = cv2.resize(test_src, dsize)







#######Original Image histogram#############
#frequency of pixels in range 0-255  and conc results
histB = cv2.calcHist([img],[0],None,[256],[0,256])
histG = cv2.calcHist([img],[1],None,[256],[0,256])
histR = cv2.calcHist([img],[2],None,[256],[0,256])
# conc all histograms
con = np.concatenate((histB, histG,histR))

######Image to be tested from infeed buffer#####
testhistB = cv2.calcHist([test],[0],None,[256],[0,256])
testhistG = cv2.calcHist([test],[1],None,[256],[0,256])
testhistR = cv2.calcHist([test],[2],None,[256],[0,256])
testCon = np.concatenate((testhistB, testhistG,testhistR))



plt.plot(con)
plt.plot(testCon)

plt.show()
#####################################################################
####################SIMILARITY EVALUATION############################
###################Infeed image  VS  Original########################
#####################################################################
####Euclidean distance
integral=0
for i in range(len(con)):
  EuclidianDistance= ((con[i]-testCon[i])**2)
  integral= integral + EuclidianDistance
else:
  ED_Result=integral**.5
  print("Euclidian distance")
  print(ED_Result)
  
  
######L1 distance
integral=0
for i in range(len(con)):
  L1= abs(con[i]-testCon[i])
  integral= integral + L1
    
else:
  L1_Result=integral
  print("L1 distance")
  print(L1_Result)
    
#####X2 distance Hellinger kernel 
integral=0
for i in range(len(con)):
  x2= ((con[i]-testCon[i])**2)/(con[i]+testCon[i])
  integral= integral + x2
    
else:
  x2_Result=integral
  print("x2 distance")
  print(x2_Result)
  
#####Hellinger kernel 
integral=0
for i in range(len(con)):
  HK= ((con[i]*testCon[i])**.5)
  integral= integral + HK
    
else:
  HK_Result=integral
  print("Hellinger Kernel distance")
  print(HK_Result)

#cv2.imshow('from infeed',test)
##cv2.imshow('Original',img)
cv2.waitKey(0)
