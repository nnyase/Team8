import argparse
import os
from backgroundRemoval import generateBackgroundMasks
from computeRetrieval import saveBestKmatches
from utils.managePKLfiles import store_in_pkl, read_pkl
from mapk import mapkL
from metricsEval import load_images_from_folder, performance_accumulation_pixel, dir_path, metrics
from generateDescriptors import computeColorDescriptors, computeTextureDescriptors
from hist2Doptimized import Create2DHistogram
from textDetection import detectTextBoxes
from meanIou import evaluateTextBoxes



def parse_args():
    parser = argparse.ArgumentParser(description= 'Descriptor generation')
    parser.add_argument('-bbddDir', '--BBDD_dir', type=str, help='Path of bbdd images')
    parser.add_argument('-qDir', '--query_dir', type=str, help='Path of query images')
    parser.add_argument('-dDir', '--descriptor_dir', type=str, default = "./descriptors/", help='Path where descriptors will be saved')
    parser.add_argument('-dColor', '--color_des', type=str, default = "no", help='Indicate if color descriptor has to be used')
    parser.add_argument('-dTexture', '--texture_des', type=str, default = "no", help='Indicate if texture descriptor has to be used')
    parser.add_argument('-numFeatures', '--num_features', type=str, default = "<>", help='Number of features for each block')
    parser.add_argument('-iLvlTexture', '--levels_texture', type=str, default = "<0>", help='Number of levels for multiresolution in texture descriptors')
    parser.add_argument('-tType', '--texture_type', type=str, default = "all", help='Texture type used to compute descriptors')
    parser.add_argument('-tBox', '--text_boxes', default = "no", type = str, help='Indicate if text boxes has to be detected')
    parser.add_argument('-tBoxDir', '--text_boxes_dir', default = "./textBoxes/", type = str, help='Path where detected text boxes will be saved')
    parser.add_argument('-rDir', '--results_dir', type=str, default = "./results/", help='Path where retrieval results will be saved')
    parser.add_argument('-rK', '--result_k', type=int, default = 10, help= "Number of predictions saved for each image in results")
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
    
# Generate color descriptors taking into account the folder management
def genAndStoreColorDescriptors(color_space, hist_type, bins_2d, levels, descriptor_dir, images_dir, database_name, 
                           maskFolder = None, background_func = None, textBoxes = None, multiple_paintings = "no"):
                        
                    
        
    # Create folder
    pathDifferentHist = "level_" + str(levels) + "/" + hist_type + "_bins_" + str(bins_2d) + "/"
    if not(maskFolder is None):
        folderName = color_space + "_" + background_func + "/" + pathDifferentHist
    else:
        folderName = color_space + "/" + pathDifferentHist
        
    folderPath = descriptor_dir + database_name + "/" + folderName
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        
        computeColorDescriptors(images_dir, folderPath, color_space, bins_2d, Create2DHistogram, levels, 
                           backgroundMaskDir = maskFolder, textBoxes = textBoxes, 
                           multipleImages= multiple_paintings)
        
        print(database_name, " descriptors generated!")
                
# Generate texture descriptors taking into account the folder management
def genAndStoreTextureDescriptors(textureTypes, levels, num_features, descriptor_dir, images_dir, database_name, 
                           maskFolder = None, background_func = None, textBoxes = None, multiple_paintings = "no"):
    
    
    for textureType in textureTypes:  
        for level in levels:
            
            for num in num_features:
            
                   
                
                namePath = "levels_" + str(level) + "/" + "features_" + str(num) + "/"
                
                # Create folder
                if not(maskFolder is None):
                    folderName = textureType + "_" + background_func + "/"
                else:
                    folderName = textureType + "/" 
                    
                folderPath = descriptor_dir + database_name + "/" + folderName + namePath
                
                if not os.path.exists(folderPath):
                    os.makedirs(folderPath)
                    
                    computeTextureDescriptors(images_dir, folderPath, textureType, level, num,
                                       backgroundMaskDir = maskFolder, textBoxes = textBoxes, 
                                       multipleImages= multiple_paintings)
                    
                    print(database_name, " descriptors generated!")
            

def mainProcess():
    # All available posibilities
    color_space = "cielab"
    hist_type = "2D"
    distance_func = "l1"
    background_func = "method3"
    bins_2d = 20
    levels = 3
    
    texture_types = ["lbp", "dct", "hog", "wavelet"]
    
    # Get args
    args = parse_args()
    
    if args.texture_type != "all":
        if args.texture_type in texture_types:
            texture_types = [args.texture_type]
        else:
            print("Not a valid texture type!")
            return
    
    # Get list arguments
    map_k_values = [int(k) for k in args.map_k_values[1:-1].split(",")]
    num_features = [int(k) for k in args.num_features[1:-1].split(",")]
    levels_texture = [int(k) for k in args.levels_texture[1:-1].split(",")]
    
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
    
    if args.color_des == "yes":
    
        # Generate color the descriptors of BBDD if they are not already generated
        genAndStoreColorDescriptors(color_space, hist_type, bins_2d, levels, args.descriptor_dir, 
                               args.BBDD_dir, "BBDD")
        
        # Generate color the descriptors of query if they are not generated
        genAndStoreColorDescriptors(color_space, hist_type, bins_2d, levels, args.descriptor_dir, 
                               args.query_dir, queryName, maskFolder, background_func, textBoxes, 
                               args.multiple_paintings)
    
    if args.texture_des == "yes":
    
        # Generate texture the descriptors of BBDD if they are not already generated
        genAndStoreTextureDescriptors(texture_types, levels_texture, num_features, args.descriptor_dir, 
                               args.BBDD_dir, "BBDD")
        
        # Generate texture the descriptors of query if they are not generated
        genAndStoreTextureDescriptors(texture_types, levels_texture, num_features, args.descriptor_dir, 
                               args.query_dir, queryName, maskFolder, background_func, textBoxes, 
                               args.multiple_paintings)
    
    
    # Compute retrieval       
    # Get descriptor folders
    if args.color_des == "yes":
        
        pathDifferentHist = "level_" + str(levels) + "/" + hist_type + "_bins_" + str(bins_2d) + "/"
        pathBBDDdescriptors = args.descriptor_dir + "BBDD/" + color_space + "/" + pathDifferentHist
        if args.background_rem == "no":
            pathQdescriptors = args.descriptor_dir + queryName + "/" + color_space + "/" + pathDifferentHist
        else:
            pathQdescriptors = args.descriptor_dir + queryName + "/" + color_space + "_" + background_func + "/" + pathDifferentHist
    
        # Compute results using distance function
            
        # Create folder
        if args.background_rem == "no":
            folderName =  color_space + "_" + distance_func + "/" + pathDifferentHist
        else:
            folderName =  color_space + "_" + distance_func + "_" + background_func + "/" + pathDifferentHist
            
        folderPath = args.results_dir + queryName + "/" + folderName
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
    
            # Compute result
            result = saveBestKmatches(pathBBDDdescriptors, pathQdescriptors, args.result_k, distance_func)
            
            # Store results
            store_in_pkl(folderPath + "result.pkl", result)
            
            print("Retrieval done!")
    
    if args.texture_des == "yes":
        
        for textureType in texture_types:
            
            for level in levels_texture:
                
                for num in num_features:
                
                       
                    
                    namePath = "levels_" + str(level) + "/" + "features_" + str(num) + "/"
                    pathBBDDdescriptors = args.descriptor_dir + "BBDD/" + textureType + "/" + namePath
                    if args.background_rem == "no":
                        pathQdescriptors = args.descriptor_dir + queryName + "/" + textureType + "/" + namePath
                    else:
                        pathQdescriptors = args.descriptor_dir + queryName + "/" + textureType + "_" + background_func + "/" + namePath
                    
                    # Compute results using distance function
                        
                    # Create folder
                    if args.background_rem == "no":
                        folderName =  textureType + "_" + distance_func + "/" + namePath
                    else:
                        folderName =  textureType + "_" + distance_func + "_" + background_func + "/" + namePath
                        
                    folderPath = args.results_dir + queryName + "/" + folderName
                    
                    if not os.path.exists(folderPath):
                        os.makedirs(folderPath)
                    
                    
                        # Compute result
                        result = saveBestKmatches(pathBBDDdescriptors, pathQdescriptors, args.result_k, distance_func)
                        
                        # Store results
                        store_in_pkl(folderPath + "result.pkl", result)
                        
                        print("Retrieval done!")
    
    # Compute retrieval evaluation if there is GT result
    if args.gt_result != "None":
        # Read GT 
        gtResults = read_pkl(args.gt_result)
        
        if args.color_des == "yes":
                
            # Get result file path
            pathDifferentHist = "level_" + str(levels) + "/" + hist_type + "_bins_" + str(bins_2d) + "/"
            if args.background_rem == "no":
                pathResults = args.results_dir + queryName + "/" + color_space + "_" + distance_func + "/" + pathDifferentHist + "/result.pkl"
            else:
                pathResults = args.results_dir + queryName + "/" + color_space + "_" + distance_func + "_" + background_func + "/" + pathDifferentHist + "/result.pkl"
    
            # Read prediction results
            predictedResults = read_pkl(pathResults)
        
            for k in map_k_values:
                # Compute mapk evaluation
                mapkValue = mapkL(gtResults, predictedResults, k)
                
                # Print results
                print("Color space: ", color_space, ", distance func: ", distance_func, ", Background_Rem: ", 
                      args.background_rem, ", hist_type: ", hist_type, ", numBins: ", bins_2d, ", levels: ", levels, 
                      ", MAP%", k, " score is: ", mapkValue)
        
        if args.texture_des == "yes":
            
            for textureType in texture_types:
                
                for level in levels_texture:
                    
                    for num in num_features:
                    
                           
                        
                        namePath = "levels_" + str(level) + "/" + "features_" + str(num) + "/"
                        # Get result file path
                        if args.background_rem == "no":
                            pathResults = args.results_dir + queryName + "/" + textureType + "_" + distance_func + "/" + namePath + "/result.pkl"
                        else:
                            pathResults = args.results_dir + queryName + "/" + textureType + "_" + distance_func + "_" + background_func + namePath + "/result.pkl"
                
                        # Read prediction results
                        predictedResults = read_pkl(pathResults)
                    
                        for k in map_k_values:
                            # Compute mapk evaluation
                            mapkValue = mapkL(gtResults, predictedResults, k)
                            
                            # Print results
                            print("Texture type: ", textureType, ", levels: ", level, ", num features: ", num, ", distance func: ", distance_func, ", Background_Rem: ", 
                                  args.background_rem, ", MAP%", k, " score is: ", mapkValue)

if __name__ == "__main__":

    #input_dir = "../../WEEK1/BBDD/"
    #output_dir = "./descriptors/"
    #mask_dir = 'None'
    
    mainProcess()