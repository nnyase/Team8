import argparse
import os
from generateDescriptors import computeDescriptors
from backgroundRemoval import generateBackgroundMasks
from computeRetrieval import saveBestKmatches
from utils.managePKLfiles import store_in_pkl, read_pkl
from ml_metrics import mapk
from metricsEval import load_images_from_folder, performance_accumulation_pixel, dir_path, metrics



def parse_args():
    parser = argparse.ArgumentParser(description= 'Descriptor generation')
    parser.add_argument('-bbddDir', '--BBDD_dir', type=str, help='Path of bbdd images')
    parser.add_argument('-qDir', '--query_dir', type=str, help='Path of query images')
    parser.add_argument('-dDir', '--descriptor_dir', type=str, help='Path where descriptors will be saved')
    parser.add_argument('-rDir', '--results_dir', type=str, help='Path where retrieval results will be saved')
    parser.add_argument('-disF', '--distance_func', type=str, help= "Distance function to use to compute similarities")
    parser.add_argument('-rK', '--result_k', type=int, help= "Number of predictions saved for each image in results")
    parser.add_argument('-c', '--color_space', type=str, help='Color space that will be used')
    parser.add_argument('-bRem', '--background_rem', default = "no", type = str, help='Method to remove the background of images')
    parser.add_argument('-maskDir', '--mask_dir', default="None", type=str, help='Path where background mask will be saved')
    parser.add_argument('-mapK', '--map_k_values', default="<1,5>", type=str, help='Which values of k use to evaluate using MAP')
    parser.add_argument('-gtR', '--gt_result',, type=str, help='Ground-truth result of query')
    parser.add_argument('-gtM', '--gt_masks', default="None", type = str, help="Path where ground-truth masks are")
    


    return parser.parse_args()


def mainProcess():
    # All available posibilities
    color_spaces = ["rgb","hsv","cielab", "cieluv", "ycbcr"]
    distance_funcs = ["euclidean", "l1", "x2", "hellinger", "histIntersect", "cosSim"]
    background_funcs = ["method1", "method2", "method3"]
    
    
    # Get args
    args = parse_args()
    
    # Check args
    if args.color_space != "all":
        if args.color_space in color_spaces:
            color_spaces = [args.color_space]
        else:
            print("Not a valid color space!")
            return
        
    if args.distance_func != "all":
        if args.distance_func in distance_funcs:
            distance_funcs = [args.distance_func]
        else:
            print("Not a valid distance function!")
            return
        
    if args.background_rem == "no":
        background_funcs = []
    elif args.background_rem != "all":
        if args.background_rem in distance_funcs:
            background_funcs = [args.background_rem]
        else:
            print("Not a valid background removal function!")
            return
        
    map_k_values = [int(k) for k in args.map_k_values[1:-1].split(",")]
        
    queryName = args.query_dir.split('/')[-2]
        
    maskFolders = []
        
    # Generate background masks
    for background_func in background_funcs:
        
        # Create folder
        folderName = background_func + "/"
        folderPath = args.mask_dir + queryName + "/" + folderName
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
        maskFolders.append(folderPath)
        
        # Generate masks
        generateBackgroundMasks(args.query_dir, folderPath, background_func)
        
        print("Masks generated!")
    
    # Evaluate masks    
    for i, maskFolder in enumerate(maskFolders):
        
        print(background_func[i], " generated mask evaluation: ")
        
        # Load files
        act = dir_path(args.gt_masks)
        pred = dir_path(maskFolder)
        actual_imgs=load_images_from_folder(act)
        predicted_imgs = load_images_from_folder(pred)
        
        # Calculate TP, FN, FP
        TP, FN, FP = performance_accumulation_pixel(actual=actual_imgs, predicted=predicted_imgs)
        
        # Calculate Precision, Recall, F1
        metrics(TP,FN,FP)
        
        
    
    # Generate the descriptors of BBDD if they are not already generated
    for color_space in color_spaces:
        
        # Create folder
        folderName = color_space + "/"
        folderPath = args.descriptor_dir + "BBDD" + "/" + folderName
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
            
            computeDescriptors(args.BBDD_dir, folderPath, color_space)
            
            print("BBDD descriptors generated!")
    
    
    # Generate the descriptors of query if they are not generated
    if args.background_rem == "no":

        for color_space in color_spaces:
            
            # Create folder
            folderName = color_space + "/"
            folderPath = args.descriptor_dir + queryName + "/" + folderName
            
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
                
                computeDescriptors(args.query_dir, folderPath, color_space)
                
                print("Query descriptors generated!")
    else:
        # For every background removal method used
        for i, maskFolder in enumerate(maskFolders):
            for color_space in color_spaces:
                
                # Create folder
                folderName = color_space + "_" + background_funcs[i] + "/"
                folderPath = args.descriptor_dir + queryName + "/" + folderName
                
                if not os.path.exists(folderPath):
                    os.makedirs(folderPath)
                    
                    computeDescriptors(args.query_dir, folderPath, color_space, True, maskFolder)
                    
                    print("Query descriptors generated!")
    
    
    # Compute retrieval
    if args.background_rem == "no":
        # Compute result for every descriptor color space
        for color_space in color_spaces:
            
            # Get descriptor folders
            pathBBDDdescriptors = args.descriptor_dir + "/BBDD/" + color_space + "/"
            pathQdescriptors = args.descriptor_dir + queryName + "/" + color_space + "/"
            
            # Compute results using each distance function
            for disFunc in distance_funcs:
                
                # Create folder
                folderName =  color_space + "_" + disFunc + "/"
                folderPath = args.results_dir + queryName + "/" + folderName
                
                if not os.path.exists(folderPath):
                    os.makedirs(folderPath)
                    
                
                    # Compute result
                    result = saveBestKmatches(pathBBDDdescriptors, pathQdescriptors, args.result_k, disFunc)
                    
                    # Store results
                    store_in_pkl(folderPath, result)
                    
                    print("Retrieval done!")
    else:
        
        
        for background_func in background_funcs:
            # Compute result for every descriptor color space
            for color_space in color_spaces:
                
                # Get BBDD descriptor folder
                pathBBDDdescriptors = args.descriptor_dir + "/BBDD/" + color_space + "/"
                
                # Get query descriptor folder
                pathQdescriptors = args.descriptor_dir + queryName + "/" + color_space + "_" + background_func + "/"
                
                # Compute results using each distance function
                for disFunc in distance_funcs:
                    
                    # Create folder
                    folderName =  color_space + "_" + disFunc + "_" + background_func + "/"
                    folderPath = args.results_dir + queryName + "/" + folderName
                    
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath)
                        
                    
                        # Compute result
                        result = saveBestKmatches(pathBBDDdescriptors, pathQdescriptors, args.result_k, disFunc)
                        
                        # Store results
                        store_in_pkl(folderPath, result)
                        
                        print("Retrieval done!")
    
    # Compute retrieval evaluation
    
    # Read GT 
    gtResults = read_pkl(args.gt_result)
    
    if args.background_rem == "no":
        for disFunc in distance_funcs:
            for color_space in color_spaces:
                
                # Get result file path
                pathResults = args.results_dir + queryName + "/" + color_space + "_" + disFunc + "/result.pkl"
    
                # Read prediction results
                predictedResults = read_pkl(pathResults)
            
                for k in map_k_values:
                    # Compute mapk evaluation
                    mapkValue = mapk(gtResults, predictedResults, k)
                    
                    # Print results
                    print("Color space: ", color_space, ", distance func: ", disFunc, ", MAP%", 
                          k, " score is: ", mapkValue)
    else:
        
        for background_func in background_funcs:
            for disFunc in distance_funcs:
                for color_space in color_spaces:
                    
                    # Get result file path
                    pathResults = args.results_dir + queryName + "/" + color_space + "_" + disFunc + "_" + background_func + "/result.pkl"
        
                    # Read prediction results
                    predictedResults = read_pkl(pathResults)
                
                    for k in map_k_values:
                        # Compute mapk evaluation
                        mapkValue = mapk(gtResults, predictedResults, k)
                        
                        # Print results
                        print("Background removal method: ", background_func, "Color space: ", color_space, ", distance func: ", 
                              disFunc, ", MAP%", k, " score is: ", mapkValue)

if __name__ == "__main__":

    #input_dir = "../../WEEK1/BBDD/"
    #output_dir = "./descriptors/"
    #mask_dir = 'None'
    
    mainProcess()
    