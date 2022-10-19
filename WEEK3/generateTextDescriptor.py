from hog import createHoGdescriptor
from dctDescriptors import createDCTdescritor
from wavelet import createWaveletDescriptor
from lbpHist import createLBPhistogram

import cv2

def generateTextureDescriptors(image):
    
    # LBP-hist
    d1 = createLBPhistogram(image, 10, 1, 255 )
    # HoG
    d2 = createHoGdescriptor(image, 255, 10)
    # DCT
    d3 = createDCTdescritor(image, 10, 255)
    # Wavelet
    d4 = createWaveletDescriptor(image)

    

imageP = "../../WEEK2/qsd1_w2/00000.jpg"
image = cv2.imread(imageP)

generateTextureDescriptors(image)