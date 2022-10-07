import ml_metrics
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# Task 6  Evaluation using MAP@K of binary masks
# Actual  Correct binary Mask given
# Predicted Query 2 binary mask results 

actual = [cv2.imread(file).tolist() for file in tqdm(glob.glob("qsd2_w1/*.png"))]
predicted = [cv2.imread(file).tolist() for file in tqdm(glob.glob("WEEK1/q2_result/*.png"))]


metric = ml_metrics.mapk(actual=actual,
                      predicted=predicted, k=1)

print("MAP@K Score: {:.4f}% ({}/{})".format(metric*100,
      int(len(actual)*metric), len(predicted)))



