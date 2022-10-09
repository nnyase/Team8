from backgroundRemoval import get_binary_mask, get_binary_mask2, get_binary_mask3
import os
import cv2
from matplotlib import pyplot as plt

path = "../../WEEK1/qsd2_w1/"

for file in os.listdir(path):
    if file[-4:] == ".jpg":
        image = cv2.imread(path+file)
        
        mask1 = get_binary_mask(image)
        mask2 = get_binary_mask2(image)
        mask3 = get_binary_mask3(image)