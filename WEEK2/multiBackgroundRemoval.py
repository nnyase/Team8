import sys
import cv2
import numpy as np
import random
from scipy.ndimage import label
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description= 'Generate Binary Masks')
    parser.add_argument('-iDir', '--input_dir', type=str, help='Path of images')
    parser.add_argument('-oDir', '--output_dir', type=str, help='Path of output images')
    parser.add_argument('-m', '--method',type=str, help ='Method')
    return parser.parse_args()

#METHOD 1
#Convert our image into HSV color space
#Perform simple thresholding to create a map using Numpy based on Saturation and Value
#Combine the map from S and V into a final mask
#Determine the foreground and background based on the combined mask
#Reconstruct original image by combining extracted foreground and background

def get_binary_mask(myimage):
    
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S (saturation) and remove any value that is less than 100
    s = myimage_hsv[:,:,1]
    s = np.where(s < 100, 0, 1) 
 
    # Take V (saturation) and remove any value that is more than 50
    v = myimage_hsv[:,:,2]
    v = np.where(v > 50, 0, 1)
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground= cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
    binary_mask = cv2.bitwise_not(background)
 
    # Post process the mask and return
    return postProcessMask(binary_mask[:,:,0])


# Giving a binary mask, it removes all not connected background areas with the most external
# component
def postProcessMask(binary_mask):
    checked = np.zeros(binary_mask.shape, dtype = np.uint8)
    background = np.zeros(binary_mask.shape, dtype = np.uint8) 
    
    queue = [(0,0)]
    checked[0,0] = 255
    background[0,0] = 255
    
    while len(queue) > 0:
        i, j = queue.pop()
        
        if i + 1 < binary_mask.shape[0]:
                
                if checked[i+1,j] == 0:
                    checked[i+1,j] = 255
                    if binary_mask[i+1,j] == 0:
                        background[i + 1, j] = 255
                    
                        queue.append((i+1,j))
                    
        if i - 1 > 0:
            if checked[i-1,j] == 0:
                checked[i-1,j] = 255
                if binary_mask[i-1,j] == 0:
                    background[i - 1, j] = 255
                
                    queue.append((i-1,j))
        
        if j + 1 < binary_mask.shape[1]:
                
                if checked[i,j+1] == 0:
                    checked[i,j+1] = 255
                    if binary_mask[i,j+1] == 0:
                        background[i, j+1] = 255
                    
                        queue.append((i,j+1))
                    
        if j - 1 > 0:
            if checked[i,j-1] == 0:
                checked[i,j-1] = 255
                if binary_mask[i,j-1] == 0:
                    background[i, j-1] = 255
                
                    queue.append((i,j-1))
        
    bg = 255- background
    return bg
                
        
                
def generateBackgroundMasks(imagesPath, masksPath):
    """ This functions creates the masks of the background of images and stores them in the output folder.
    

    Parameters
    ----------
    imagesPath : str
        Input images path.
    masksPath : str
        Output folder path, where mask will be saved.
    method : str
        Background removal method to use. Options are: "method1", "method2", "method3"

    Returns
    -------
    None.

    """
    
    # Get file names of the database
    files = os.listdir(imagesPath)

    for file in files:
            # Check if it is an image 
            if file[-4:] == ".jpg":

                # Read the image
                image = cv2.imread(imagesPath + file)

                filename = masksPath + file[:-4] + ".png"
                binary_mask =  get_binary_mask(image)
                    
                # Save the mask
                cv2.imwrite(filename, binary_mask)


    
#generateBackgroundMasks(imagesPath='WEEK2/qsd2_w2/',masksPath='WEEK2/bgRemovalImages2/', method = 'method3')

def segment_on_dt(img):
    dt = cv2.distanceTransform(img, 2, 3) # L2 norm, 3x3 mask
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)[1]
    lbl, ncc = label(dt)

    lbl[img == 0] = lbl.max() + 1
    lbl = lbl.astype(np.int32)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lbl)
    lbl[lbl == -1] = 0
    return lbl

#image = 'WEEK2/qsd2_w2/00000.jpg'

def detectPainting(path):
    image = cv2.imread(path)
    print(type(image))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    img = 255 - img # White: objects; Black: background

    ws_result = segment_on_dt(img)
    # Colorize
    height, width = ws_result.shape
    ws_color = np.zeros((height, width, 3), dtype=np.uint8)
    lbl, ncc = label(ws_result)
    for l in range(1, ncc + 1):
        a, b = np.nonzero(lbl == l)
        if img[a[0], b[0]] == 0: # Do not color background.
            continue
        rgb = [random.randint(0, 255) for _ in range(3)]
        ws_color[lbl == l] = tuple(rgb)
    return img

def detectPaintings(imagesPath,outputPath):
    """ This functions detects paintings from images using given path
    

    Parameters
    ----------
    imagesPath : str
        Input images path.

    Returns
    -------
    None.

    """
    
    # Get file names of the database
    files = os.listdir(imagesPath)
    

    for file in (files):
            # Check if it is an image 
            if file[-4:] == ".jpg":
                print(file)

                # Read the image
                filename = output_path + file[:-4] + ".png"
                img = detectPainting(file)
                    
                # Save the img
                cv2.imwrite(filename, img)

#detectPaintings('WEEK2/qsd2_w2/')


if __name__ == "__main__":
    args = parse_args()
    img_path = args.input_dir
    output_path = args.output_dir
    method = args.method

    if method == 'method1':
        generateBackgroundMasks(img_path,output_path)
    elif method =='method2':
        detectPaintings(img_path,output_path)
    





