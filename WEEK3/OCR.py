import cv2
import pytesseract
import os
import numpy as np
from utils.managePKLfiles import read_pkl
from utils.distanceTextMetrics import getDistance2Strings
import textdistance
from textDDetection import detectTextBoxes
from mapk import mapkL
from getBiggestAreasContours import getBiggestContours


pytesseract.pytesseract.tesseract_cmd = r'C:/Users/inigo/anaconda3/envs/py38/Library/bin/tesseract.exe'

'''
                This script contains three main functions :
                    
                 1.-   getTextDescriptorsFromImages:
                         By giving a set of images it will extract the text on them based in
                         the boxes coordinates obtained from detectTextBoxes function  
                         , it stores the extracted text in NPY or txt files in the outputPah
                         
                 2.-   getTextDescriptorsFomTxtFiles:
                         By giving a set of text files it extracts the painting name from the string
                         and stores it in a NPY  or txt file 
                     
                 3.-   saveBestKmatches_Text:
                         This function computes all the similarities between the database and query painting names
                             using the distance function given and returns k best matches for every query image
                

 '''

def getPaintName(inputPath):   
    """ This function is used to give format to text strings in the 
        getTextDescriptorsFomTxtFiles fucntion 
    Parameters
    ----------
    inputPath : path where .txt file is located

    Returns
    -------
    final : final string 
    """
    my_file = open(inputPath,"r")
    final= " "
    content = my_file. read()
    content = str.strip (content)
    content1 = content.split(",")   
    l = len(content1)
    
    if l ==2:
        final=content1[2]
        lfinal=len(final)
        final = final[:lfinal-2]
        final = final[2:]

    elif l ==3:
        final=content1[1]
        final = final[2:]
        
    my_file. close()
    return final


def getPainterName(inputPath):
    """ This function is used to give format to text strings in the 
        getTextDescriptorsFomTxtFiles fucntion  (NOT USED FOR NOW)
    Parameters
    ----------
    inputPath : path where .txt file is located

    Returns
    -------
    final : final string 
    """
    my_file = open(inputPath,"r")
    content = my_file. read()
   
    line = content.split(",")
    painterTemp=line[0]
    painterName = painterTemp[2:-1]
    
    my_file. close()
    return painterName
def extractTextOnce(img,BBox):
    
    roi = img[BBox[1]:BBox[3],BBox[0]:BBox[2]]
    #roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    #_, roiT = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(roi)
    #print(text)
    return text

def computeTextDescriptorsFromImages(inputPath, outputPath, textBoxes,
                       backgroundMaskDir = None, multipleImages = "no"):
    
    """ This function extract text from an image using detectTextBoxes function
        By giving the path where images are located it will store the detected text
        in the putput parameter
    Parameters
    ----------
    inputPath : path where images files are located
    outputPath : path where NPY files will be stored
    textBoxes  : list of list of list of detected boxes 
    backgroundMaskDir : background mask direction, if there is no background values should be None
    multipleImages: "yes" if the image can have more than one painting, otherwise "no"
    -------
    """
    
    # Iterate files
    for file in os.listdir(inputPath):
        if file[-4:] == ".jpg":
            
            imageNum = int(file[:-4].split("_")[-1])
            # Read image
            image = cv2.imread(inputPath + file)
            
            # Get mask
            if backgroundMaskDir is None:
                # Take into account every pixel
                mask = np.zeros(image.shape[:2], dtype = np.uint8) + 255
            else:
                mask = cv2.imread(backgroundMaskDir + file[:-4] + ".png", cv2.IMREAD_GRAYSCALE)

            
            if multipleImages != "no":
                
                boxes = getBiggestContours(mask)
                # Empty mask
                mask = np.zeros(image.shape[:2], dtype = np.uint8) + 255
            
            for i, box in enumerate(boxes):
                xMinP, yMinP, xMaxP, yMaxP = box
                
                paintingNew =  image[yMinP: yMaxP + 1, xMinP: xMaxP + 1]
                    
                # Extract text from image BBoxs
                text=extractTextOnce(paintingNew,textBoxes[imageNum][i])
                # Erase n/ char
                textFinal=str.strip(text)
                
                descriptorPath = outputPath + file[:-4] + "_" + str(i) + ".npy"
                np.save(descriptorPath, textFinal)
                

def computeTextDescriptorsFomTxtFiles(inputPath,outputPath):
    """ This function extract painting name from the strings stored in .txt files and store them in NPY 
    Parameters
    ----------
    inputPath : string where .txt files are located
    outputPath : string where NPY files will be stored
    -------
    """
    
    
    # Iterate files
    for file_text in os.listdir(inputPath):
        if file_text[-4:] == ".txt":
            # Extract paitning name
            #textFiles_Paintname= getPaintName(inputPath+file_text)
            
            # Extract painter name
            textFiles_Paintername= getPainterName(inputPath+file_text)
            
            # Save as descriptors
            np.save(outputPath + file_text[:-4] + ".npy"  ,textFiles_Paintername)
     





def loadNPY(inputPath):
    """ This function reads all NPY files inside a folder and prints them 
    Parameters
    ----------
    inputPath : string where npy are located

    Returns
    -------
    resultsFromFiles : All data in 1 vector
    """
    resultsFromFiles = []
    i=0
    # Iterate files
        # Extract text from .txt files
    for file_text in os.listdir(inputPath):
        if file_text[-4:] == ".npy":
            result=np.load(inputPath+file_text)
            if not result:
                result= ""
                print(result)
            else:
                print(result)
            resultsFromFiles.append(result)
            i=i+1
    return resultsFromFiles





def saveBestKmatches(bbddDescriptorsPath, qDescriptorsPath,npy_OR_txt, k, distanceFunc):
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
    result : list of lists of lists (int)
        The best k matches for each image in the query. The k matches are sorted from
        the most similar to the least one.
    """
    
    # Compute number of images in each set
    numBBDD = len(os.listdir(bbddDescriptorsPath))
    numQ = len(os.listdir(qDescriptorsPath))
    
    
    # Get distance function

        
    # Init result list
    result = []
    
    # For every image in query
    for i, fileQ in enumerate(os.listdir(qDescriptorsPath)):
        
        # Get descriptor path
        descriptors_Q1_Path = qDescriptorsPath + fileQ
        str1=np.load(descriptors_Q1_Path)
        # Create list of distances
        distances = np.array([-1.]*numBBDD)
        
        # For every image in BBDD
        for j, fileBBDD in enumerate(os.listdir(bbddDescriptorsPath)):
            
            # Get descriptor path
            descriptors_DDBB_Path = bbddDescriptorsPath + fileBBDD
            str2=np.load(descriptors_DDBB_Path)
           
            # Calculate distance, if empty add an empty string
            str11=str(str1)
            str22=str(str2)

            distance = getDistance2Strings(str11,str22,distanceFunc)
            # Save distance
            distances[j] = distance

        # Sort the distances and get k smallest values indexes
        sortedIndexes = np.argsort(distances)
        
        # Save results in the list
        if int(fileQ[:-4].split("_")[-1]) == 0:
            result.append([sortedIndexes[:k].tolist()])
        else:
            result[-1].append(sortedIndexes[:k].tolist())
    
    return result

"""
outputPath_qsd1 = './textDescriptors/qsd1_w2/'
outputPath_BBDD = './textDescriptors/BBDD/'
inputPath_qsd1 = './denoisedImages/optimized/qsd1_w3/'
inputPath_BBDD = '../../WEEK1/BBDD/'
gt_results = "../../WEEK3/qsd1_w3/"+"gt_corresps.pkl"


# Calculate descriptors
#computeTextDescriptorsFomTxtFiles(inputPath_BBDD,outputPath_BBDD)
#computeTextDescriptorsFromImages(inputPath_qsd1, outputPath_qsd1,False)

# Compute matches
for i in range(1,36):
    print("Distance function: ", i)
    predictedResults=saveBestKmatches(outputPath_BBDD, outputPath_qsd1, 10, i)#6)
    gtResults = read_pkl(gt_results)
    # print("GT:")
    # print(gtResults)
    # print("----------------------------------")
    # print("Best K matches")
    # print(predictedResults)
    
    # Evaluate results
    mapkValue = mapkL(gtResults, predictedResults, 1)
    print(mapkValue)
    mapkValue = mapkL(gtResults, predictedResults, 5)
    print(mapkValue)
    #print(results)


#cv2.waitKey(27)
"""