import cv2
import numpy as np

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
def contourLocation(cntrn):
    minX = np.min(cntrn[:,:,0])
    maxX = np.max(cntrn[:,:,0])
    minY = np.min(cntrn[:,:,1])
    maxY = np.max(cntrn[:,:,1])
  
    return (minX, minY, maxX, maxY)
    
def getBiggestContours(img):
    #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #edges= cv2.Canny(gray, 50,200)
    contours, hierarchy= cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    
    foundContours = list()
    
    minArea = img.shape[0]/10 * img.shape[1]/10
    for contour in sorted_contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            foundContours.append(contourLocation(contour))
            # 2 Max contours
            if len(foundContours) > 1:
                break
        else:
            break

    if len(foundContours)>1:
        if foundContours[0][0] > foundContours[1][0]:
            
            foundContours[0], foundContours[1] = foundContours[1], foundContours[0]
    
    return foundContours
    

    
