import numpy as np
import cv2
from tqdm import tqdm
import glob
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description= 'Generate Precision, Recall and F1 for list of images')
    parser.add_argument('-aDir', '--actual_input_dir', type=str, help='Path of Actual images')
    parser.add_argument('-pDir', '--predicted_input_dir', type=str, help='Path of predicted images')
    return parser.parse_args()

# Calculates the F1, P, Recall and also the TP,FN,FP of a list of images
def read_images_from_dir(actual,predicted):
    actual_imgs = [cv2.imread(file) for file in tqdm(glob.glob(actual))]
    predicted_imgs = [cv2.imread(file) for file in tqdm(glob.glob(predicted))]
    return actual_imgs, predicted_imgs



def performance_accumulation_pixel(actual, predicted):
    """
    Function to compute different performance indicators for images
    (True Positive, False Positive, False Negative, True Negative) 
    at the pixel level.

    Parameters
    Takes in a list of images. 
    Actual: Ground-Truth
    target: Images to be compared to Ground-Truth

    Return

    TP: True Positive
    FP: False Positive
    TN: True Negatice
    FN: False Negative
    """ 
    TP_list, FN_list, FP_list = [],[],[]
    for image_actual, image_pred in zip(actual, predicted):
        mask_actual = image_actual[:, :, 0]#/255
        mask_predicted = image_pred[:, :, 0]#/255
        TP = np.sum(cv2.bitwise_and(mask_actual, mask_predicted)) / 255
        FN = np.sum(cv2.bitwise_and(mask_actual, cv2.bitwise_not(mask_predicted))) / 255
        FP = np.sum(cv2.bitwise_and(cv2.bitwise_not(mask_actual), mask_predicted)) / 255
        TP_list.append(TP)
        FN_list.append(FN)
        FP_list.append(FP)

    return TP_list,FN_list,FP_list

def metrics(TP_list,FN_list,FP_list): 
    """ 
    Function to compute precision, recall, f1
    with a list of TP,FN,FP scores

    Parameters
    list of TP,FN,FP

    Return
    p: precision
    r: recall
    f1: f1 score

    """
    p_list, r_list, f1_list = [], [], []
    for TP,FN,FP in zip(TP_list,FN_list,FP_list):
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * ((P*R)/(P+R))
        p_list.append(P)
        r_list.append(R)
        f1_list.append(F1)
    p, r, f1 = np.sum(p_list)/len(TP_list), np.sum(r_list)/len(TP_list), np.sum(f1_list)/len(TP_list)
    
    return print("Precision: ", p,"Recall: ", r,"F1 score:", f1)

#TP,FP,TN = performance_accumulation_pixel(actual,predicted)
#p,r,f1 = metrics(TP,FP,TN)
def dir_path(string):
    if os.path.isdir(string):
        print(string," is a path")
        return string
    else:
        return "Not a path"


def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder)):
        if filename[-4:] == ".png":
            img = cv2.imread(os.path.join(folder,filename))
            images.append(img)
    return images




if __name__ == "__main__":
    args = parse_args()
    act = dir_path(args.actual_input_dir)
    pred = dir_path(args.predicted_input_dir)
    actual_imgs=load_images_from_folder(act)
    predicted_imgs = load_images_from_folder(pred)
    TP, FN, FP = performance_accumulation_pixel(actual=actual_imgs, predicted=predicted_imgs)
    metrics(TP,FN,FP)

