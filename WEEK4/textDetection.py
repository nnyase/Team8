import cv2
import numpy as np
import os
from utils.managePKLfiles import store_in_pkl
from utils.getBiggestAreasContours import getBiggestContours

def detectText(image):
    """ This functions returns the BBox of the text box in the input image given.
    

    Parameters
    ----------
    image : numpy array (uint8)
        image of one painting (without background) in BGR color space.

    Returns
    -------
    list
        box coordinates.

    """
    # Get grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get saturation (S)
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageS = imageHSV[:,:,1]
    
    # Threshold saturation with small values
    imageSmallS = imageS < 20
    
    
    # Morphological gradients
    kernel = np.ones([3,3])
    dilateImage = cv2.dilate(imageGray, kernel, 1)
    erodeImage = cv2.erode(imageGray, kernel, 1)
    morphologicalGradient = dilateImage - erodeImage
    
    # Gradients of only small saturation
    morphologicalGradient = morphologicalGradient * imageSmallS 
    
    # Binarize
    _, binary = cv2.threshold(morphologicalGradient, 70, 255, cv2.THRESH_BINARY)
    
    # Set edges to 0
    binary[0,:] = 0
    binary[binary.shape[0]-1,:] = 0
    binary[:,0] = 0
    binary[:, binary.shape[1]-1] = 0
    
    # Put the letters together
    kernel = np.ones([1,int(image.shape[1]/25)])
    binaryClose1 = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, kernel)
    
    # Remove noise
    kernel = np.ones([5,5])
    binaryOpen = cv2.morphologyEx(binaryClose1,cv2.MORPH_OPEN, kernel)
    
    
    # Put the words together
    kernel = np.ones([1,int(image.shape[1]/5)])
    binaryClose = cv2.morphologyEx(binaryOpen,cv2.MORPH_CLOSE, kernel)
    
    # Remove noise
    kernel = np.ones([5,5])
    binaryOpen = cv2.morphologyEx(binaryClose,cv2.MORPH_OPEN, kernel)
    
    # Set edges to 0
    binaryOpen[0,:] = 0
    binaryOpen[binaryOpen.shape[0]-1,:] = 0
    binaryOpen[:,0] = 0
    binaryOpen[:, binaryOpen.shape[1]-1] = 0
    
    
    # Get biggest contour
    contour = getBiggestContour(binaryOpen)
    
    # Compute BBox
    if not (contour is None):
        # Get BBox from contour
        minX = int(np.min(contour[:,0,0]))
        maxX = int(np.max(contour[:,0,0]))
        
        minY = int(np.min(contour[:,0,1]))
        maxY = int(np.max(contour[:,0,1]))
        
        
        # Expand the bbox
        diffX = int((maxX - minX)*0.03)
        minX = max(0, minX-diffX)
        maxX = min(image.shape[1] - 1, maxX + diffX)
        
        
        diffY = int((maxY - minY)*0.30)
        minY = max(0, minY-diffY)
        maxY = min(image.shape[0] - 1, maxY + diffY)
    
    
    else:
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0

    cv2.rectangle(image,(minX,minY),(maxX,maxY),[0,0,255],2)
    
    return [minX, minY, maxX, maxY]


def getBiggestContour(binaryImage):
    """
    This function returns the bigges contour in the binary input image.

    Parameters
    ----------
    binaryImage : numpy array (uint8)
        Binary image to get contours from.

    Returns
    -------
    biggestContour : numpy array (uint8)
        Found biggest contour.

    """
    biggestContour = None
    biggestArea = -1    
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > biggestArea:
            
            biggestArea = area
            biggestContour = contour
            
            
    
    return biggestContour

def detectTextBoxes(inputPath, outputPath, multiplePaintings = "No", maskDir = None, numberOfFile = -1):
    """ This function detects the text boxes of the images that are in the input path.
    

    Parameters
    ----------
    inputPath : str
        Images path.
    outputPath : str
        Path to save the box detections.
    multiplePaintings : str, optional
        Indication if the images can have more than one painting ("yes" or "no"). The default is "no".
    maskDir : str, optional
        Background mask path. If the image has multiple painting, indicate here the background masks path. The default is None.
    numberOfFile, int, optional
        Number of the file that will be in the result file: text_boxes{numberOfFile}.pkl. Otherwise will be: text_boxes.pkl.

    Returns
    -------
    results : numpy array (int)
        detected text boxes.

    """
    # Init box list
    results = []
    
    # Iterate files
    for file in os.listdir(inputPath):
        if file[-4:] == ".jpg":
            
            # Init image box list
            resultsImage = []
            # Read image
            image = cv2.imread(inputPath + file)
            
            # Check if there are more than one painting
            if multiplePaintings == "yes":
                # Read mask
                mask = cv2.imread(maskDir + file[:-4] + ".png", cv2.IMREAD_GRAYSCALE)
                # Get multiple paintings boxes
                boxes = getBiggestContours(mask)
                
                paintings = []
                # Get cropped painting images
                for box in boxes:
                    xMin, yMin, xMax, yMax = box
                    paintings.append(image[yMin:yMax + 1, xMin:xMax + 1])
            
            else:
                paintings = [image]
            
            # For each painting detect text box
            for i, painting in enumerate(paintings):
                
                detection = detectText(painting)
                
                # If the paintings are cropped, get real positions
                if multiplePaintings == "yes":
                    xMin, yMin, _, _ = boxes[i] 
                    
                    detection[0] += xMin
                    detection[1] += yMin
                    detection[2] += xMin
                    detection[3] += yMin
                
                resultsImage.append(detection)
                
            results.append(resultsImage)
    
    # Store result
    if numberOfFile == -1:
        store_in_pkl(outputPath + "text_boxes.pkl", results)
    else:
        store_in_pkl(outputPath + "text_boxes" + str(numberOfFile) + ".pkl", results)
    
    
    
    return results
    
            

# inputPath = "../../WEEK2/qsd1_w2/"
# outputPath = "./textBoxes/"

# detectTextBoxes(inputPath, outputPath , 1)
