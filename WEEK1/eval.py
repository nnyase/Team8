from os import SCHED_OTHER
import numpy as np
import cv2
from tqdm import tqdm
import glob

# Calculates the F1, P, Recall and also the TP,FN,FP of a list of images


actual = [cv2.imread(file) for file in tqdm(glob.glob("qsd2_w1/*.png"))]
predicted = [cv2.imread(file) for file in tqdm(glob.glob("WEEK1/q2_result/*.png"))]


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
        mask_actual = image_actual[:, :, 0]/255
        mask_predicted = image_pred[:, :, 0]/255
        TP = np.sum(cv2.bitwise_and(mask_actual, mask_predicted))
        FN = np.sum(cv2.bitwise_and(mask_actual, cv2.bitwise_not(mask_predicted)))
        FP = np.sum(cv2.bitwise_and(cv2.bitwise_not(mask_actual), mask_predicted))
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

    return p, r, f1

TP,FP,TN = performance_accumulation_pixel(actual,predicted)
p,r,f1 = metrics(TP,FP,TN)

print("Precision: ", p,"Recall: ", r,"F1 score:", f1)
#print("True Positive: ", sum(TP)/len(TP),"False Positive: ",sum(FP)/len(FP),"True Negative: ",sum(TN)/len(TN))
#print(TP)
