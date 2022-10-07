from ml_metrics import mapk
from utils.managePKLfiles import read_pkl



if __name__ == "__main__":

    # Args
    pathPredictions = "./result.pkl"
    pathGT = "../../WEEK1/qsd1_w1/gt_corresps.pkl"
    k = 3


    # Read results
    gtResults = read_pkl(pathGT)
    predictedResults = read_pkl(pathPredictions)

    # Compute mapk evaluation
    mapkValue = mapk(gtResults, predictedResults, 3)
    
    # Print results
    print("MAP%" + k + " score is: " + mapkValue)


