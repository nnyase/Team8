import cv2
import numpy as np
from textDetection import detectTextBoxes
from meanIou import evaluateTextBoxes


def topHatOrBlackHat(image):

    """
    This function decide which morphological operations is the best between topHat and blackHat
    
    Parameters
    ----------
    image : numpy array

    Returns
    -------
    tophat_img or blackhat_img

    """

    # Getting the kernel to be used in Top-Hat
    filterSize =(3,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize)
    
    # Applying the Top-Hat operation
    tophat_img = cv2.morphologyEx(image, 
                              cv2.MORPH_TOPHAT,
                              kernel)

    # Applying the Top-Hat operation
    blackhat_img = cv2.morphologyEx(image, 
                              cv2.MORPH_BLACKHAT,
                              kernel)

    # Compare the number of white pixel in tophat_img and blackhat_img
    if (tophat_img > 200).sum() > (blackhat_img > 200).sum() :
        return tophat_img
    else:
        return blackhat_img


def detectTextBox(pathOfImage):

    """
    This function detect the text in an image
    
    Parameters
    ----------
    pathOfImage : str path of the image

    Returns
    -------
    results : numpy array (int)
        detected text boxes.

    """

    # Load the image
    img = cv2.imread(pathOfImage)
    # Convert the image in grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Use the best between topHat and blackHat
    result = topHatOrBlackHat(gray)

    # List to store position of contours
    xlist = []
    ylist = []
    wlist = []
    hlist = []


    contours, hier = cv2.findContours(result,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cntr in contours:

        x,y,w,h = cv2.boundingRect(cntr)
        if w > h and w < 0.9*img.shape[1] and w > 0.2*img.shape[1] and h < 1/8*img.shape[0]:
            
            xlist.append(x)
            ylist.append(y)
            wlist.append(w)
            hlist.append(h)

    if(len(xlist) ==  0):
        # No detection of text
        return [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]

    # Check which rectangle has his center of mass nearest the vertical center line
    center = [ (xlist[i] + wlist[i])/2 for i in range(len(xlist))]
    position = min(range(len(center)), key=lambda i: abs(center[i]- int(img.shape[1]/2)))


    # Draw the text box
    cv2.rectangle(img, (xlist[position], ylist[position]), (xlist[position]+wlist[position], ylist[position]+hlist[position]), (0, 0, 255), 2)


    #Optional
    cv2.imshow('test',img)
    cv2.waitKey(0)

    return [np.array([xlist[position], ylist[position]]), np.array([xlist[position], ylist[position] + hlist[position]]), np.array([xlist[position] + wlist[position], ylist[position] + hlist[position]]), np.array([xlist[position] + wlist[position], ylist[position]])]
    


inputPath = 'WEEK2/qsd1_w2/'
outputPath = 'WEEK2/textBoxes/'
detectTextBoxes(inputPath, outputPath, 2)

evaluateTextBoxes('WEEK2/textBoxes/text_boxes2.pkl', "WEEK2/qsd1_w2/text_boxes.pkl")

#detectTextBox('teste/00014.jpg')