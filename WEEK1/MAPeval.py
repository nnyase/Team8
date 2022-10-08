from ml_metrics import mapk
from utils.managePKLfiles import read_pkl



if __name__ == "__main__":

    # Args
    pathPredictions = "./results/qsd1/"
    pathGT = "../../WEEK1/qsd1_w1/gt_corresps.pkl"
    kv = [1,3]
    
    distanceFuncsStr = ["euclidean", "l1", "x2", "hellinger", "histIntersect", "cosSim"]
    colorSpaces = ["rgb","hsv","cielab", "cieluv", "ycbcr"]
    
    # Read GT 
    gtResults = read_pkl(pathGT)
    
    for disF in distanceFuncsStr:
        for colorSpace in colorSpaces:
            
            # Get result file path
            pathResults = pathPredictions + colorSpace + "_" + disF + "/result.pkl"

            # Read prediction results
            predictedResults = read_pkl(pathResults)
        
            for k in kv:
                # Compute mapk evaluation
                mapkValue = mapk(gtResults, predictedResults, k)
                
                # Print results
                print("Color space: ", colorSpace, ", distance func: ", disF, ", MAP%", 
                      k, " score is: ", mapkValue)


