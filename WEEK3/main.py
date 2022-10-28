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
from OCR import computeTextDescriptorsFromImages, computeTextDescriptorsFromTxtFiles
from utils.distanceTextMetrics import TEXT_DISTANCE_FUNCS
from task4 import saveBestKmatchesNew
from denoise import denoiseImages

def parse_args():
    parser = argparse.ArgumentParser(description= 'Descriptor generation')
    parser.add_argument('-bbddDir', '--BBDD_dir', type=str, help='Path of bbdd images')
    parser.add_argument('-qDir', '--query_dir', type=str, help='Path of query images')
    parser.add_argument('-dDir', '--descriptor_dir', type=str, default = "./descriptors/", help='Path where descriptors will be saved')
    parser.add_argument('-noise', '--noise', type=str, default = "no", help='Indicate if there is noise in images')
    parser.add_argument('-dType', '--des_type', type=str, default = "<color,texture,text>", help='Indicate the descriptor combination')
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
        
    folderPath = descriptor_dir + database_name + "/color/" + folderName
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        
        computeColorDescriptors(images_dir, folderPath, color_space, bins_2d, Create2DHistogram, levels, 
                           backgroundMaskDir = maskFolder, textBoxes = textBoxes, 
                           multipleImages= multiple_paintings)
        
        print(database_name, " descriptors generated!")
                
# Generate texture descriptors taking into account the folder management
def genAndStoreTextureDescriptors(textureType, levels, num_features, descriptor_dir, images_dir, database_name, 
                           maskFolder = None, background_func = None, textBoxes = None, multiple_paintings = "no"):
    
     
    namePath = "levels_" + str(levels) + "/" + "features_" + str(num_features) + "/"
    
    # Create folder
    if not(maskFolder is None):
        folderName = textureType + "_" + background_func + "/"
    else:
        folderName = textureType + "/" 
        
    folderPath = descriptor_dir + database_name + "/texture/" + folderName + namePath
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        
        computeTextureDescriptors(images_dir, folderPath, textureType, levels, num_features,
                           backgroundMaskDir = maskFolder, textBoxes = textBoxes, 
                           multipleImages= multiple_paintings)
        
        print(database_name, " descriptors generated!")

# Generate texture descriptors taking into account the folder management
def genAndStoreTextDescriptors(descriptor_dir, images_dir, database_name, textBoxes = None,
                           maskFolder = None, background_func = None, multiple_paintings = "no"):
    
    
    # Create folder
    if not(maskFolder is None):
        folderName = background_func + "/"
    else:
        folderName = "" 
        
    folderPath = descriptor_dir + database_name + "/text/" + folderName
    
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        
        if database_name == "BBDD":
            computeTextDescriptorsFromTxtFiles(images_dir, folderPath)
        else:
            computeTextDescriptorsFromImages(images_dir, folderPath, textBoxes,
                               backgroundMaskDir = maskFolder, 
                               multipleImages= multiple_paintings)
        
        print(database_name, " descriptors generated!")
   
# Function to compute retrieval using the saved descriptors
def computeRetrieval(args, queryName, des_combination, distance_func_text, distance_func_text_index, distance_func_vector, levels,
                     num_features, texture_type, hist_type, bins_2d, color_space, background_func = None):
    # Output path PKL file
    outputPath = args.results_dir + queryName + "/"
    
    # Set values to None
    pathBBDDdescriptorsText = None
    pathQdescriptorsText = None
    pathBBDDdescriptorsTexture = None
    pathQdescriptorsTexture = None
    pathBBDDdescriptorsColor = None
    pathQdescriptorsColor = None
    
    for des in des_combination:
        outputPath = outputPath + des + "_"
        
        
        if des == "text":
            outputPath = outputPath + "_" + distance_func_text + "_"
            
            pathBBDDdescriptorsText = args.descriptor_dir + "BBDD/text/"
            if args.background_rem != "no": 
                pathQdescriptorsText = args.descriptor_dir + queryName + "/text/" + background_func + "/"
            else:
                pathQdescriptorsText = args.descriptor_dir + queryName + "/text/"
                
            
            
            
        elif des == "texture":
            outputPath = outputPath + "_" + distance_func_vector + "_"
            
            namePath = "levels_" + str(levels) + "/" + "features_" + str(num_features) + "/"
            pathBBDDdescriptorsTexture = args.descriptor_dir + "BBDD/" + texture_type + "/" + namePath
            if args.background_rem == "no":
                pathQdescriptorsTexture = args.descriptor_dir + queryName + "/texture/" + texture_type + "/" + namePath
            else:
                pathQdescriptorsTexture = args.descriptor_dir + queryName + "/texture/" + texture_type + "_" + background_func + "/" + namePath
                
            
        elif des == "color":
            outputPath = outputPath + "_" + distance_func_vector + "_"
            
            pathDifferentHist = "level_" + str(levels) + "/" + hist_type + "_bins_" + str(bins_2d) + "/"
            pathBBDDdescriptorsColor = args.descriptor_dir + "BBDD/color/" + color_space + "/" + pathDifferentHist
            if args.background_rem == "no":
                pathQdescriptorsColor = args.descriptor_dir + queryName + "/color/" + color_space + "/" + pathDifferentHist
            else:
                pathQdescriptorsColor = args.descriptor_dir + queryName + "/color/" + color_space + "_" + background_func + "/" + pathDifferentHist
            
        
    outputPath = outputPath[:-1] + "/"
    
    
    
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
        
    results = saveBestKmatchesNew(bbddDescriptorsPathText = pathBBDDdescriptorsText, bbddDescriptorsPathColor = pathBBDDdescriptorsColor, 
                        bbddDescriptorsPathTexture = pathBBDDdescriptorsTexture, qDescriptorsPathText = pathQdescriptorsText, 
                        qDescriptorsPathColor = pathQdescriptorsColor, qDescriptorsPathTexture = pathQdescriptorsTexture, 
                        distanceFuncText = distance_func_text_index, distanceFuncColor = distance_func_vector, 
                        distanceFuncTexture = distance_func_vector, weightText = 1/2, weightColor = 1/6, weightTexture = 1/3)
    
    store_in_pkl(outputPath + "result.pkl", results)
        
    print("Retrieval done!")
    
    return outputPath
       

def mainProcess():
    # Noise reduction best method
    noise_method = "optimized"
    
    # Color best combination
    color_space = "cielab"
    hist_type = "2D"
    distance_func_vector = "l1"
    background_func = "method3"
    bins_2d = 20
    levels = 3
    
    # Texture best combination
    texture_type = "hog" # ["lbp", "dct", "hog", "wavelet"]
    num_features = 160
    
    # Text
    distance_func_text_index = 15
    distance_func_text = TEXT_DISTANCE_FUNCS[distance_func_text_index]
    
    
    descriptor_available_types = ["text", "color", "texture"]
    
    # Get args
    args = parse_args()
    
    
    # Get list arguments
    map_k_values = [int(k) for k in args.map_k_values[1:-1].split(",")]
    des_combination = args.des_type[1:-1].split(",")
    
    
    # Check if descriptor combination is valid
    for des_type in des_combination:
        if des_type not in descriptor_available_types:
            print("Not a valid descriptor type!")
            return
    
    # Get query name
    queryName = args.query_dir.split('/')[-2]
    
    # If images contain noise remove it
    if args.noise == "yes":
        
        outputPath = "./denoisedImages/" + noise_method + "/" + queryName + "/"
        
        # See if the images are already denoised
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
            
            denoiseImages(args.query_dir, outputPath, noise_method)
            print("Images denoised!")
        
        args.query_dir = outputPath
        
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
    
    # Create color descriptors if there are in the combination
    if "color" in des_combination:
    
        # Generate color the descriptors of BBDD if they are not already generated
        genAndStoreColorDescriptors(color_space, hist_type, bins_2d, levels, args.descriptor_dir, 
                               args.BBDD_dir, "BBDD")
        
        # Generate color the descriptors of query if they are not generated
        genAndStoreColorDescriptors(color_space, hist_type, bins_2d, levels, args.descriptor_dir, 
                               args.query_dir, queryName, maskFolder, background_func, textBoxes, 
                               args.multiple_paintings)
    
    # Create texture descriptors if there are in the combination
    if "texture" in des_combination:
    
        # Generate texture the descriptors of BBDD if they are not already generated
        genAndStoreTextureDescriptors(texture_type, levels, num_features, args.descriptor_dir, 
                               args.BBDD_dir, "BBDD")
        
        # Generate texture the descriptors of query if they are not generated
        genAndStoreTextureDescriptors(texture_type, levels, num_features, args.descriptor_dir, 
                               args.query_dir, queryName, maskFolder, background_func, textBoxes, 
                               args.multiple_paintings)
    
    # Create text descriptors if there are in the combination
    if "text" in des_combination:
        
        # Generate text descriptors of BBDD if they are not already generated
        genAndStoreTextDescriptors(args.descriptor_dir, args.BBDD_dir, "BBDD")
        
        # Generate text descriptors of query if they are not generated
        genAndStoreTextDescriptors(args.descriptor_dir, args.query_dir, queryName, 
                                   textBoxes, maskFolder, background_func, args.multiple_paintings)
    
    
    # Compute retrieval       
    resultsPath = computeRetrieval(args, queryName, des_combination, distance_func_text, distance_func_text_index, 
                                   distance_func_vector, levels, num_features, texture_type, hist_type, bins_2d, 
                                   color_space, background_func)
    
    # Compute retrieval evaluation if there is GT result
    if args.gt_result != "None":
        
        # Read GT 
        gtResults = read_pkl(args.gt_result)
        
        
        # Read prediction results
        predictedResults = read_pkl(resultsPath + "result.pkl")
        
        if args.noise == "yes":
            print("Noise removal method: ", noise_method)
        
        print("Combination:")
        print(args.des_type)
        for des in des_combination:
            if des == "text":
                print("Text distance function: ", distance_func_text)
            elif des == "texture":
                print("Vector distance function: ", distance_func_vector)
            elif des == "color":
                print("Vector distance function: ", distance_func_vector)
                
            print(des)
            
        
        for k in map_k_values:
            # Compute mapk evaluation
            mapkValue = mapkL(gtResults, predictedResults, k)
            
            # Print results
            print("MAP%", k, " score is: ", mapkValue)
        

if __name__ == "__main__":

    #input_dir = "../../WEEK1/BBDD/"
    #output_dir = "./descriptors/"
    #mask_dir = 'None'
    
    mainProcess()