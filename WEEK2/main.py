import argparse
import os
from backgroundRemoval import generateBackgroundMasks
from computeRetrieval import saveBestKmatches
from utils.managePKLfiles import store_in_pkl, read_pkl
from mapk import mapkL
from metricsEval import load_images_from_folder, performance_accumulation_pixel, dir_path, metrics
from generateDescriptors import computeDescriptors
from hist2Doptimized import Create2DHistogram
from hist3Doptimized import Create3DHistogram
from textDetection import detectTextBoxes
from meanIou import evaluateTextBoxes



def parse_args():
    parser = argparse.ArgumentParser(description= 'Descriptor generation')
    parser.add_argument('-bbddDir', '--BBDD_dir', type=str, help='Path of bbdd images')
    parser.add_argument('-qDir', '--query_dir', type=str, help='Path of query images')
    parser.add_argument('-dDir', '--descriptor_dir', type=str, default = "./descriptors/", help='Path where descriptors will be saved')
    parser.add_argument('-hType', '--hist_type', type=str, default = "all", help='Histogram type to use: 2D, 3D or all')
    parser.add_argument('-bins2D', '--num_bin_2D', default = "<>", type=str, help='Sequence of number of bins used for 2D histograms')
    parser.add_argument('-bins3D', '--num_bin_3D', default = "<>", type=str, help='Sequence of number of bins used for 3D histograms')
    parser.add_argument('-levels', '--multi_res_levels', default = "<0>", type = str, help='Sequence of number of multiresolution levels')
    parser.add_argument('-tBox', '--text_boxes', default = "no", type = str, help='Indicate if text boxes has to be detected')
    parser.add_argument('-tBoxDir', '--text_boxes_dir', default = "./textBoxes/", type = str, help='Path where detected text boxes will be saved')
    parser.add_argument('-rDir', '--results_dir', type=str, default = "./results/", help='Path where retrieval results will be saved')
    parser.add_argument('-disF', '--distance_func', type=str, default = "l1", help= "Distance function to use to compute similarities")
    parser.add_argument('-rK', '--result_k', type=int, default = 10, help= "Number of predictions saved for each image in results")
    parser.add_argument('-c', '--color_space', type=str, default = "cielab", help='Color space that will be used')
    parser.add_argument('-mPaintings', '--multiple_paintings', type=str, default = "no", help='Indicate if in the images could be multiple paintings')
    parser.add_argument('-bRem', '--background_rem', default = "no", type = str, help='Indicate if the query images have background')
    parser.add_argument('-maskDir', '--mask_dir', default="None", type=str, help='Path where background mask will be saved')
    parser.add_argument('-mapK', '--map_k_values', default="<1,5>", type=str, help='Which values of k use to evaluate using MAP')
    parser.add_argument('-gtR', '--gt_result', default = "None", type=str, help='Ground-truth result of query')
    parser.add_argument('-gtM', '--gt_masks', default="None", type = str, help="Path where ground-truth masks are")
    parser.add_argument('-gtT', '--gt_text_boxes', default="None", type = str, help="Path where ground-truth text boxes are")
    


    return parser.parse_args()

def genAndStoreBackgroundMasks(background_func, mask_dir, query_dir, queryName):
    # Create folder
    folderName = background_func + "/"
    folderPath = mask_dir + queryName + "/" + folderName
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    
        # Generate masks
        generateBackgroundMasks(query_dir, folderPath, background_func)
        print("Masks generated!")
    
    maskFolder = folderPath

    return maskFolder

def evaluateBackgroundMasks(gt_masks, maskFolder, background_func):
    
    print(background_func, " generated mask evaluation: ")
    
    # Load files
    act = dir_path(gt_masks)
    pred = dir_path(maskFolder)
    actual_imgs=load_images_from_folder(act)
    predicted_imgs = load_images_from_folder(pred)
    
    # Calculate TP, FN, FP
    TP, FN, FP = performance_accumulation_pixel(actual=actual_imgs, predicted=predicted_imgs)
    
    # Calculate Precision, Recall, F1
    metrics(TP,FN,FP)
    
def genAndStoreDescriptors(color_spaces, hist_types, bins_2d, bins_3d, levels, descriptor_dir, images_dir, database_name, 
                           maskFolder = None, background_func = None, textBoxes = None, multiple_paintings = "no"):
                        
    for color_space in color_spaces:
        
        for hist_type in hist_types:
            if hist_type == "2D":
                bins = bins_2d
                histGenFunc = Create2DHistogram
            else:
                bins = bins_3d
                histGenFunc = Create3DHistogram
            
            for level in levels:
                for num in bins:
                    
        
                    # Create folder
                    pathDifferentHist = "level_" + str(level) + "/" + hist_type + "_bins_" + str(num) + "/"
                    if not(maskFolder is None):
                        folderName = color_space + "_" + background_func + "/" + pathDifferentHist
                    else:
                        folderName = color_space + "/" + pathDifferentHist
                        
                    folderPath = descriptor_dir + database_name + "/" + folderName
                    
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath)
                        
                        computeDescriptors(images_dir, folderPath, color_space, num, histGenFunc, level, 
                                           backgroundMaskDir = maskFolder, textBoxes = textBoxes, 
                                           multipleImages= multiple_paintings)
                        
                        print(database_name, " descriptors generated!")
                
            
            

def mainProcess():
    # All available posibilities
    color_spaces = ["rgb","hsv","cielab", "cieluv", "ycbcr"]
    hist_types = ["2D", "3D"]
    distance_funcs = ["euclidean", "l1", "x2", "hellinger", "cosSim"]
    background_func = "method3"
    
    
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

    if args.hist_type != "all":
        if args.hist_type in hist_types:
            hist_types = [args.hist_type]
        else:
            print("Not a valid histogram type!")
            return
    
    # Get list arguments
    bins_2d = [int(k) for k in args.num_bin_2D[1:-1].split(",")]
    bins_3d =[int(k) for k in args.num_bin_3D[1:-1].split(",")]
    levels = [int(k) for k in args.multi_res_levels[1:-1].split(",")]
    map_k_values = [int(k) for k in args.map_k_values[1:-1].split(",")]
    
    # Get query name
    queryName = args.query_dir.split('/')[-2]
        
        
    # Generate background masks
    if args.background_rem != "no":
        
        maskFolder = genAndStoreBackgroundMasks(background_func, args.mask_dir, args.query_dir, queryName)
    
    else:
        maskFolder = None
    
    # Evaluate mask if there is GT 
    if args.gt_masks != "None":
        
        evaluateBackgroundMasks(args.gt_masks, maskFolder, background_func)
            
    # Generate textBoxes
    if args.text_boxes == "yes":
        textBoxes = detectTextBoxes(args.query_dir, args.text_boxes_dir, args.multiple_paintings, maskFolder)
    else:
        textBoxes = None
        
    # Evaluate text boxes if there is GT
    if args.gt_text_boxes != "None":
        
        evaluateTextBoxes(args.gt_text_boxes, args.text_boxes_dir + "text_boxes.pkl")
    
    # Generate the descriptors of BBDD if they are not already generated
    genAndStoreDescriptors(color_spaces, hist_types, bins_2d, bins_3d, levels, args.descriptor_dir, 
                           args.BBDD_dir, "BBDD")
    
    # Generate the descriptors of query if they are not generated
    genAndStoreDescriptors(color_spaces, hist_types, bins_2d, bins_3d, levels, args.descriptor_dir, 
                           args.query_dir, queryName, maskFolder, background_func, textBoxes, 
                           args.multiple_paintings)
    
    
    
    # Compute retrieval       
    # Compute result for every descriptor color space
    for color_space in color_spaces:
        
        for hist_type in hist_types:
            if hist_type == "2D":
                bins = bins_2d
            else:
                bins = bins_3d
            
            for level in levels:
                for num in bins:
        
                    # Get descriptor folders
                    pathDifferentHist = "level_" + str(level) + "/" + hist_type + "_bins_" + str(num) + "/"
                    pathBBDDdescriptors = args.descriptor_dir + "BBDD/" + color_space + "/" + pathDifferentHist
                    if args.background_rem == "no":
                        pathQdescriptors = args.descriptor_dir + queryName + "/" + color_space + "/" + pathDifferentHist
                    else:
                        pathQdescriptors = args.descriptor_dir + queryName + "/" + color_space + "_" + background_func + "/" + pathDifferentHist
            
                    # Compute results using each distance function
                    for disFunc in distance_funcs:
                        
                        # Create folder
                        if args.background_rem == "no":
                            folderName =  color_space + "_" + disFunc + "/" + pathDifferentHist
                        else:
                            folderName =  color_space + "_" + disFunc + "_" + background_func + "/" + pathDifferentHist
                            
                        folderPath = args.results_dir + queryName + "/" + folderName
                        
                        if not os.path.exists(folderPath):
                            os.makedirs(folderPath)
                        
                    
                            # Compute result
                            result = saveBestKmatches(pathBBDDdescriptors, pathQdescriptors, args.result_k, disFunc)
                            
                            # Store results
                            store_in_pkl(folderPath + "result.pkl", result)
                            
                            print("Retrieval done!")
   
    
    # Compute retrieval evaluation if there is GT result
    if args.gt_result != "None":
        # Read GT 
        gtResults = read_pkl(args.gt_result)
        
        for disFunc in distance_funcs:
            for color_space in color_spaces:
                for hist_type in hist_types:
                    if hist_type == "2D":
                        bins = bins_2d
                    else:
                        bins = bins_3d
                    
                    for level in levels:
                        for num in bins:
                
                            # Get result file path
                            pathDifferentHist = "level_" + str(level) + "/" + hist_type + "_bins_" + str(num) + "/"
                            if args.background_rem == "no":
                                pathResults = args.results_dir + queryName + "/" + color_space + "_" + disFunc + "/" + pathDifferentHist + "/result.pkl"
                            else:
                                pathResults = args.results_dir + queryName + "/" + color_space + "_" + disFunc + "_" + background_func + "/" + pathDifferentHist + "/result.pkl"
                
                            # Read prediction results
                            predictedResults = read_pkl(pathResults)
                        
                            for k in map_k_values:
                                # Compute mapk evaluation
                                mapkValue = mapkL(gtResults, predictedResults, k)
                                
                                # Print results
                                print("Color space: ", color_space, ", distance func: ", disFunc, ", Background_Rem: ", 
                                      args.background_rem, ", hist_type: ", hist_type, ", numBins: ", num, ", levels: ", level, 
                                      ", MAP%", k, " score is: ", mapkValue)

if __name__ == "__main__":

    #input_dir = "../../WEEK1/BBDD/"
    #output_dir = "./descriptors/"
    #mask_dir = 'None'
    
    mainProcess()