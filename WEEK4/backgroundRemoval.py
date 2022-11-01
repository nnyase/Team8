import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

from utils.metricsEval import performance_accumulation_pixel, metrics, read_images_from_dir

# METHOD 1
# Obtain the binary mask of an image
def get_binary_mask(image):

    # Converting into graysacle
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding after Gaussian filtering

    # Gaussian filterning is used to remove noises in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Otsu's thresholding gives an arbitrary chosen value as a threshold
    # Image with only two distinct image values (eg: background and painting)
    # where the histogram would only consist of two peaks, 
    # a good threshold would be in the middle of those two values
    ret,binary_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Not operator
    binary_mask = cv2.bitwise_not(binary_mask)

    # Convert 255 into 1 to have a binary mask
    #binary_mask[binary_mask == 255] = 1 

    #return the binary mask as numpy array
    return binary_mask



# METHOD 2
# 1. Convert our image into Greyscale
# 2. Perform simple thresholding to build a mask for the foreground and background
# 3. Determine the foreground and background based on the mask
# 4. Reconstruct original image by combining foreground and background

def get_binary_mask2(myimage):
    # First Convert to Grayscale
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    # Trunc thresholding - Truncate Thresholding ( THRESH_TRUNC ) ·
    #The destination pixel is set to the threshold ( thresh ), if the source pixel value is greater than the threshold.
    ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
    # Threshold to remove background from grayscaled image
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
    # Threshold to extract the background from the grayscaled image
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Combine the background and foreground to obtain our final image with background extracted
    finalimage = background+foreground
    bg = cv2.bitwise_not(background)
    return bg

#METHOD 3
#Convert our image into HSV color space
#Perform simple thresholding to create a map using Numpy based on Saturation and Value
#Combine the map from S and V into a final mask
#Determine the foreground and background based on the combined mask
#Reconstruct original image by combining extracted foreground and background

def get_binary_mask3(myimage):
    
    
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S (saturation) and remove any value that is less than 100
    s = myimage_hsv[:,:,1]
    s = np.where(s < 100, 0, 1) 
 
    # Take V (value) and remove any value that is more than 50
    v = myimage_hsv[:,:,2]
    v = np.where(v > 80, 0, 1)
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
    
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones([10,10]))
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground= cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
    binary_mask = cv2.bitwise_not(background)
 
    # Post process the mask and return
    return postProcessMask(binary_mask[:,:,0])
    #return binary_mask[:,:,0]


# Method 4
def get_binary_mask4(myimage):
    # Convert to grayscale
    myimageG = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian Blur and compute differences
    hpfG = myimageG - cv2.GaussianBlur(myimageG, (21, 21), 3)+127
    
    # Get maximum difference areas
    _, foreground1 = cv2.threshold(hpfG, np.max(hpfG)*5/9, 255, cv2.THRESH_BINARY)
    _, foreground2 = cv2.threshold(hpfG, 255 - np.max(hpfG)*5/9, 255, cv2.THRESH_BINARY_INV)
    
    # Maximum areas are contour of the paintings
    foreground = np.where(foreground1+foreground2>0, 255, 0)
    foreground = foreground.astype(np.uint8)
    
    # Close with vertical and horizontal kernels
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones([1,20]))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones([20,1]))
    
    # Flood fill mask
    foreground = postProcessMask(foreground)
    
    # Open to remove noise
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones([20,20]))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones([1,int(myimage.shape[1]/5)]))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones([int(myimage.shape[0]/5),1]))

    return foreground

# Giving a binary mask, it removes all not connected background areas with the most external
# component
def postProcessMask(binary_mask):
    checked = np.zeros(binary_mask.shape, dtype = np.uint8)
    background = np.zeros(binary_mask.shape, dtype = np.uint8) 
    queue = []
    for i in range(binary_mask.shape[0]):
        # Left
        queue.append((i,0))
        checked[i,0] = 255
        background[i,0] = 255
        # Right
        queue.append((i,binary_mask.shape[1]-1))
        checked[i,binary_mask.shape[1]-1] = 255
        background[i,binary_mask.shape[1]-1] = 255
        
    for i in range(binary_mask.shape[1]):
        # Top
        queue.append((0,i))
        checked[0,i] = 255
        background[0,i] = 255
        # Bot
        queue.append((binary_mask.shape[0] - 1, i))
        checked[binary_mask.shape[0] - 1, i] = 255
        background[binary_mask.shape[0] - 1, i] = 255
            
    
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
                
        
#TODO new method 
#metricsEval.py

# Method 5 : comes from team 9

def crop_img(image_array,top,bottom,left,right):
    """ Cuts off the specified amount of pixels of an image
        top,bottom,keft,right: amount of px to crop in each direction
        
    """
    height = image_array.shape[0]
    width = image_array.shape[1]
    cropped_image = image_array[int(top):int(height-bottom),int(left):int(width-right)]
    return cropped_image

def get_binary_mask5(img):
    """
    Given an image, the gradient of its grayscale version is computed (edges of the image) and they are expanded with morphological operations
    """
        
    img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #define kernels
    kernel_size_close = 20
    kernel_size_close2 = 100
    kernel_size_remove = 1500
    kernel_size_open = 70
    
    
    gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close,kernel_size_close))
    kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close2,kernel_size_close2))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_open,kernel_size_open))
    
    kernel_close_vert = cv2.getStructuringElement(cv2.MORPH_RECT,(2,kernel_size_remove))
    kernel_close_hor = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_remove,2))

    #obtain gradient of grayscale image
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, gradient_kernel)
    
    #binarise gradient
    temp, gradient_binary = cv2.threshold(gradient,30,255,cv2.THRESH_BINARY)
    mask = gradient_binary[:,:,0]
    

    #add zero padding for morphology tasks 
    padding = 1500
    mask = cv2.copyMakeBorder( mask,  padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
    
    #slight closing to increase edge size
    mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    #really wide closing in horizontal and vertical directions
    temp1 = mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_vert)
    
    temp2 = mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_hor)

    #the mask will be the intersection
    mask = cv2.bitwise_and(temp1, temp2)
    
    #small opening and closing
    mask = mask =cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)
    mask = crop_img(mask ,padding,padding,padding,padding)

    mask = mask.astype(np.uint8)

    return mask



def generateBackgroundMasks(imagesPath, masksPath, method):
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

                if method == "method1":
                    binary_mask =  get_binary_mask(image)
                elif method == "method2":
                    binary_mask =  get_binary_mask2(image)
                elif method == "method3":
                    binary_mask =  get_binary_mask3(image)
                elif method == "method4":
                    binary_mask =  get_binary_mask4(image)
                elif method == "method5":
                    binary_mask =  get_binary_mask5(image)
                    
                # Save the mask
                cv2.imwrite(filename, binary_mask)

# For generating masks

# generateBackgroundMasks('WEEK4/denoisedImages/optimized/qsd1_w4/', 'WEEK4/masks/qsd1_w4/method5/', "method5")

# Test background removal performance

# python3 metricsEval.py -aDir ../masks/qsd1_w4/actual -pDir ../masks/qsd1_w4/method5