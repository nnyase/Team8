import cv2
import pytesseract
import os
import numpy as np
from utils.managePKLfiles import read_pkl
from utils.distanceTextMetrics import getDistance2Strings
import textdistance
from textDDetection import detectTextBoxes
from mapk import mapkL
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/inigo/anaconda3/envs/py38/Library/bin/tesseract.exe'

'''
                This script contains three main functions :
                    
                 1.-   getTextDescriptorsFromImages:
                         By giving a set of images it will extract the text on them based in
                         the boxes coordinates obtained from detectTextBoxes function  
                         , it stores the extracted text in PNY or txt files in the outputPah
                         
                 2.-   getTextDescriptorsFomTxtFiles:
                         By giving a set of text files it extracts the painting name from the string
                         and stores it in a PNY  or txt file 
                     
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

def getTextDescriptorsFromImages(inputPath, outputPath,pny_OR_txt,createPKL):
    
    """ This function extract text from an image using detectTextBoxes function
        By giving the path where images are located it will store the detected text
        in the putput parameter
    Parameters
    ----------
    inputPath : path where images files are located
    outputPath : path where NPY files will be stored
    txt_OR_pny  : Boolean value to select how the data wil be saved 
                    True=  pny files   False= txt files
    Returns
    -------
    resultsFromBBoxes : All data in 1 vector
    """
    resultsFromBBoxes = []
    BBox=detectTextBoxes(inputPath,outputPath,createPKL, numberOfFile=2)
    # Iterate files
    i=0
    for file in os.listdir(inputPath):
        if file[-4:] == ".jpg":
            
            # Read image
            img = cv2.imread(inputPath + file)
            # Extract text from image BBoxs
            text=extractTextOnce(img,BBox[i][0])
            # Erase n/ char
            textFinal=str.strip(text)
            resultsFromBBoxes.append(textFinal)
            # save in a PNY file
<<<<<<< Updated upstream
            if pny_OR_txt :
                np.save(outputPath + file[:-4] + ".npy"  ,textFinal)
     
            else:
                file2save= open(outputPath + file[:-4] + ".txt",'w') 
                file2save.write(textFinal)
                file2save.close()
            
            
            
            
            
=======
            np.save(outputPath + file[:-4] + "_0.npy", textFinal)
>>>>>>> Stashed changes
            i=i+1
    return resultsFromBBoxes

def getTextDescriptorsFomTxtFiles(inputPath,outputPath,pny_OR_txt):
    """ This function extract painting name from the strings stored in .txt files and store them in PNY 
    Parameters
    ----------
    inputPath : string where .txt files are located
    outputPath : string where NPY files will be stored
    txt_OR_pny  : Boolean value to select how the data wil be saved 
                    True=  pny files   False= txt files
    Returns
    -------
    resultsFromFiles : All data in 1 vector
    """
    resultsFromFiles = []
    # Iterate files
    i=0
        # Extract text from .txt files
    for file_text in os.listdir(inputPath):
        if file_text[-4:] == ".txt":
            # Extract paitning name
            #textFiles_Paintname= getPaintName(inputPath+file_text)
            # Extract painter name
            textFiles_Paintername= getPainterName(inputPath+file_text)
            # Save as descriptors
<<<<<<< Updated upstream
            resultsFromFiles.append(textFiles_Paintname)
            if pny_OR_txt :
                np.save(outputPath + file_text[:-4] + ".npy"  ,textFiles_Paintname)
     
            else:
                file2save= open(outputPath + file_text[:-4] + ".txt",'w') 
                file2save.write(textFiles_Paintname)
                file2save.close()
=======
            np.save(outputPath + file_text[:-4] + ".npy"  ,textFiles_Paintername)#textFiles_Paintname)
            resultsFromFiles.append(textFiles_Paintername)#textFiles_Paintname)
>>>>>>> Stashed changes
            
            i=i+1
    #print(resultsFromFiles)
    return resultsFromFiles





def loadNPY(inputPath):
    """ This function reads all PNY files inside a folder and prints them 
    Parameters
    ----------
    inputPath : string where pny are located

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





def saveBestKmatches(bbddDescriptorsPath, qDescriptorsPath,pny_OR_txt, k, distanceFunc):
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
           
<<<<<<< Updated upstream
            # Calculate distance, 
            str11=np.array2string(str1)
            str22=np.array2string(str2)
=======
            # Calculate distance, if empty add an empty string
            str11=str(str1)
            str22=str(str2)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream




outputPath_qsd1_pny = './textDescriptors/notDenoisedImages/qsd1_w2_pny/'
outputPath_BBDD_pny = './textDescriptors/BBDD_pny/'
outputPath_qsd1_txt = './textDescriptors/qsd1_w2_txt/'
outputPath_BBDD_txt = './textDescriptors/BBDD_txt/'
inputPath_qsd1 = './qsd1_w2/'
inputPath_BBDD = './BBDD/'

inputPath_source = './denoisedImages/gaussian /qsd1_w3/'
outputPath_results = './textDescriptors/denoisedImages/nlmean/qsd1_w3_npy/'




gt_results = inputPath_qsd1+"gt_corresps.pkl"



# gtResults = read_pkl(gt_results)

#predictedResults=saveBestKmatches(outputPath_BBDD_txt, outputPath_qsd1_txt,True, 1, 3)

# print("GT:")
# print(gtResults)
# print("----------------------------------")
# print("Best K matches")
# print(predictedResults)
# mapkValue = mapkL(gtResults, predictedResults, 10)
# print(mapkValue)
# print(results)
# getTextDescriptorsFomTxtFiles(inputPath_BBDD,outputPath_BBDD,False)
#results=getTextDescriptorsFromImages(inputPath_source, outputPath_results,True,False)
#results1=loadNPY(outputPath_results)
#print(results)
# j=0
# for i in range(35):
#     predictedResults=saveBestKmatches(outputPath_BBDD_pny, outputPath_qsd1_pny,False, 5, j)
#     mapkValue = mapkL(gtResults, predictedResults, 5)
#     print(str(mapkValue)+" Num:"+str(j))
#     j=j+1
    
=======
outputPath_qsd1 = './textDescriptors/qsd1_w2/'
outputPath_BBDD = './textDescriptors/BBDD/'
inputPath_qsd1 = './denoisedImages/optimized/qsd1_w3/'
inputPath_BBDD = '../../WEEK1/BBDD/'
gt_results = "../../WEEK3/qsd1_w3/"+"gt_corresps.pkl"











# Calculate descriptors
#getTextDescriptorsFomTxtFiles(inputPath_BBDD,outputPath_BBDD)
#getTextDescriptorsFromImages(inputPath_qsd1, outputPath_qsd1,False)

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

>>>>>>> Stashed changes
#cv2.waitKey(27)
