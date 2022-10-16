# Week 2

Image retrieval using multiresolution 2D and 3D histograms. If images contain background, it has to be removed and if it contains text boxes, they have to be detected. In some images there are more than one painting.

## Usage:
``main.py`` can be used to process week 2 queries (``qsd1_w2`` and ``qsd2_w2``). ``mainNewHistW1.py`` can be used to get results of the retrieval system using new multiresolution 2D and 3D histograms in week 1 queries.

### Example

Compute ``qsd2_w2`` query set, with CIELAB color space, L1 distance function, with 2D histograms of [5,10,20,45,90] bins, with 3D histograms of [5,10,20,30] bins for [0,1,2,3] levels.

``
mainNewHistW1.py -bbddDir ../../WEEK1/BBDD/ -qDir ../../WEEK1/qsd2_w1/ -dDir ./descriptors/ -hType all -bins2D <5,10,20,45,90> -bins3D <5,10,20,30> -levels <0,1,2,3> -rDir ./results/ -disF l1 -rK 10 -c cielab -bRem yes -maskDir ./masks/ -gtR ../../WEEK1/qsd2_w1/gt_corresps.pkl -gtM ../../WEEK1/qsd2_w1/
``

Compute ``qsd1_w2`` query set, with CIELAB color space, L1 distance, with level 3 2D histograms (of20 bins) and 3D histograms (of 10 bins).

``
main.py "-bbddDir ../../WEEK1/BBDD/ -qDir ../../WEEK2/qsd1_w2/ -hType all -bins2D <20> -bins3D <10> -levels <3> -tBox yes -gtR ../../WEEK2/qsd1_w2/gt_corresps.pkl -gtT ../../WEEK2/qsd1_w2/text_boxes.pkl"
``

``
 main.py [-h] [-bbddDir BBDD_DIR] [-qDir QUERY_DIR]
               [-dDir DESCRIPTOR_DIR] [-hType HIST_TYPE] [-bins2D NUM_BIN_2D]
               [-bins3D NUM_BIN_3D] [-levels MULTI_RES_LEVELS]
               [-tBox TEXT_BOXES] [-tBoxDir TEXT_BOXES_DIR]
               [-rDir RESULTS_DIR] [-disF DISTANCE_FUNC] [-rK RESULT_K]
               [-c COLOR_SPACE] [-mPaintings MULTIPLE_PAINTINGS]
               [-bRem BACKGROUND_REM] [-maskDir MASK_DIR] [-mapK MAP_K_VALUES]
               [-gtR GT_RESULT] [-gtM GT_MASKS] [-gtT GT_TEXT_BOXES]
``

```

Options:

  -bbddDir BBDD_DIR, --BBDD_dir BBDD_DIR
                        Path of bbdd images.
						
  -qDir QUERY_DIR, --query_dir QUERY_DIR
                        Path of query images.
                        
  -dDir DESCRIPTOR_DIR, --descriptor_dir DESCRIPTOR_DIR
                        Path where descriptors will be saved. The default value is ./descriptors/.
                        
  -hType HIST_TYPE, --hist_type HIST_TYPE
                        Histogram type to use: 2D, 3D or all. The default value is all.
                        
  -bins2D NUM_BIN_2D, --num_bin_2D NUM_BIN_2D
                        Sequence of number of bins used for 2D histograms. The default value is <>.
                        
  -bins3D NUM_BIN_3D, --num_bin_3D NUM_BIN_3D
                        Sequence of number of bins used for 3D histograms. The default value is <>.
                        
  -levels MULTI_RES_LEVELS, --multi_res_levels MULTI_RES_LEVELS
                        Sequence of number of multiresolution levels. The default value is <0>  (no multiresolution).
                        
  -tBox TEXT_BOXES, --text_boxes TEXT_BOXES
                        Indicate if text boxes has to be detected: yes or no. The default value is no.
                        
  -tBoxDir TEXT_BOXES_DIR, --text_boxes_dir TEXT_BOXES_DIR
                        Path where detected text boxes will be saved. The default value is ./textBoxes/.
                        
  -rDir RESULTS_DIR, --results_dir RESULTS_DIR
                        Path where retrieval results will be saved. The default value is ./results/.
                        
  -disF DISTANCE_FUNC, --distance_func DISTANCE_FUNC
                        Distance function to use to compute similarities. The default values is l1.
                        
  -rK RESULT_K, --result_k RESULT_K
                        Number of predictions saved for each image in results. The default value is 10.
                        
  -c COLOR_SPACE, --color_space COLOR_SPACE
                        Color space that will be used: rgb, cielab, cieluv, hsv or ycrbc. The default value is cielab.
                        
  -mPaintings MULTIPLE_PAINTINGS, --multiple_paintings MULTIPLE_PAINTINGS
                        Indicate if in the images could be multiple paintings: yes or no. The default value is no.
                        
  -bRem BACKGROUND_REM, --background_rem BACKGROUND_REM
                        Indicate if the query images have background: yes or no. The default value is no.
                        
  -maskDir MASK_DIR, --mask_dir MASK_DIR
                        Path where background mask will be saved. The default value is "None".
                        
  -mapK MAP_K_VALUES, --map_k_values MAP_K_VALUES
                        Which values of k use to evaluate using MAP. The default value is <1,5>.
                        
  -gtR GT_RESULT, --gt_result GT_RESULT
                        Ground-truth result of query. The default value is "None".
                        
  -gtM GT_MASKS, --gt_masks GT_MASKS
                        Path where ground-truth masks are. The default value is "None".
                        
  -gtT GT_TEXT_BOXES, --gt_text_boxes GT_TEXT_BOXES. 
                        Path where ground-truth text boxes are. The default value is "None".

```
