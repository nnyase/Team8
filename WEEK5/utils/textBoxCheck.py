import os
import cv2
from utils.managePKLfiles import read_pkl
from matplotlib import pyplot as plt

pathPredicted = "../textBoxes/text_boxes.pkl"
tbP = read_pkl(pathPredicted)
pathGT = "../../../WEEK5/qsd1_w5/text_boxes.pkl"
tbGT = read_pkl(pathGT)

pathImages = "../../../WEEK5/qsd1_w5/" 

indexImg = 0
for file in os.listdir(pathImages):
    
    if file[-4:] == ".jpg":
        
        image = cv2.imread(pathImages + file)
        pred = tbP[indexImg]
        gt = tbGT[indexImg]

        for i, paintingPD in enumerate(pred):
            paintingGT = gt[i]
            
            for j, point in enumerate(paintingPD):
                cv2.circle(image, (int(point[0]), int(point[1])), 0, [255,0,0], 50)
                cv2.circle(image, (int(paintingGT[j][0]), int(paintingGT[j][1])), 0, [0,0,255], 50)
        plt.imshow(image)
        plt.show()
        
        
        indexImg += 1