import cv2

"""
    get2Biggercontourns
    
    This function finds two bigger contourns in an image

    Parameters
    ----------
    img : mat 
    activateDrawContours: if true draws the contourns 


    Returns
    -------
    Results : vector with  4 tuples  which are the location of the maxs an mins 
    of each contourn in the order  [minX , maxX ]  [maxY ,  minY]  


    """
def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas
def contournLocation(cntrn):
    minX = tuple(cntrn[cntrn[:,:,0].argmin()][0])
    maxX = tuple(cntrn[cntrn[:,:,0].argmax()][0])
    maxY = tuple(cntrn[cntrn[:,:,1].argmin()][0])
    minY = tuple(cntrn[cntrn[:,:,1].argmax()][0])
    X=[minX,maxX]
    Y=[minY,maxY]
  
    return X,Y
    
def get2Biggercontourns(img,activateDrawContours):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges= cv2.Canny(gray, 50,200)
    contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

    first= sorted_contours[0]
    second= sorted_contours[1]
    first_Locations= contournLocation(first)
    second_Locations= contournLocation(second)
    results= [first_Locations,second_Locations]
    if activateDrawContours:
        cv2.drawContours(img, first, -1, (0,255,0),10)
        cv2.drawContours(img, second, -1, (0,255,0),10)
    return results
    

    
