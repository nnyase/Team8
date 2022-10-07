from numpy.linalg import norm
from numpy import dot

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
# Histogram Intersection
def Hist_Intersection(con,testCon):
    HI=0
    for i in range(len(con)):
        HI += min(con[i], testCon[i])
        #print("Histogram Intersection")
        #print(HI)
    return HI

# Cosine Similarity
def Cosine_Similarity(con,testCon):
    cos_sim = dot(con, testCon)/(norm(con)*norm(testCon))
    #print("Cosine similarity")
    #print(cos_sim)
    return cos_sim