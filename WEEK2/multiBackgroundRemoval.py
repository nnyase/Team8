import cv2
import numpy as np
import random
import numpy
import random
from scipy.ndimage import label
import os
from tqdm import tqdm
import argparse

def segment_on_dt(img):
    """ This function thresholds an image after apply a 
    distance transformation. Then applys a watershed algorithm to isolate 
    the painting from the background.

   
    Parameters
     -----------

    image: cv2imread image

    Returns 
 -----------

    lbl: watershed markers
    
    
    
    """
    dt = cv2.distanceTransform(img, 2, 3) # L2 norm, 3x3 mask
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)[1]
    lbl, ncc = label(dt)

    lbl[img == 0] = lbl.max() + 1
    lbl = lbl.astype(np.int32)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lbl)
    lbl[lbl == -1] = 0
    return lbl


def detectPainting(path):
    """ This function applys the water an image after apply a 
    distance transformation. Then applys a watershed algorithm to isolate 
    the painting from the background.

    
    Parameters
    -----------
    image: cv2imread image
    
   
    Returns 
     --------

    img: mask binary image that attempts to segement background and foreground for two paintings
    
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    img = 255 - img # White: objects; Black: background

    ws_result = segment_on_dt(img)
    # Colorize the watershed result
    height, width = ws_result.shape
    ws_color = np.zeros((height, width, 3), dtype=numpy.uint8)
    lbl, ncc = label(ws_result)
    for l in range(1, ncc + 1):
        a, b = np.nonzero(lbl == l)
        if img[a[0], b[0]] == 0: # Do not color background.
            continue
        rgb = [random.randint(0, 255) for _ in range(3)]
        ws_color[lbl == l] = tuple(rgb)

    return img
    

files = 'WEEK2/qsd2_w2/'
filename = 'WEEK2/task6imgs/'

for file in os.listdir(files):
    if file[-4:] == '.jpg':
        d = detectPainting(files+file)
        file_saved = filename + file[:-4] + ".png"
        cv2.imwrite(file_saved, d)
