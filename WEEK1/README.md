# Week 1

Image retrieval using 1D histograms. If images contain background, it has to be removed.

## Usage:

`` main.py `` script can be used to process every task of week 1.
If ``"all"`` type of arguments are given, results and generated files are going to be saved in different folders. 

``
$ main.py [-h] [-bbddDir BBDD_DIR] [-qDir QUERY_DIR][-dDir DESCRIPTOR_DIR] [-rDir RESULTS_DIR] [-disF DISTANCE_FUNC] [-rK RESULT_K] [-c COLOR_SPACE] [-bRem BACKGROUND_REM] [-maskDir MASK_DIR] [-mapK MAP_K_VALUES] [-gtR GT_RESULT] [-gtM GT_MASKS]
``

```

Options:

  -bbddDir BBDD_DIR, 		Path of bbdd images.
  
  -qDir QUERY_DIR, 			Path of query images.
  
  -dDir DESCRIPTOR_DIR, 	Path where descriptors will be saved.
  
  -rDir RESULTS_DIR, 		Path where retrieval results will be saved.
  
  -disF DISTANCE_FUNC, 		Distance function to use to compute similarities. 
							Options are: "euclidean", "l1", "x2", "hellinger", "cosSim" or "all".
							If "all" is selected every function will be used.
  
  
  -rK RESULT_K, 			Number of predictions saved for each image in results.
  
  -c COLOR_SPACE, 			Color space that will be used. 
							Options are: "rgb","hsv","cielab", "cieluv", "ycbcr" or "all".
							If "all" is selected every color space will be used.
  
  -bRem BACKGROUND_REM, 	Method to remove the background of images.
							Options are: "no", "method1", "method2", "method3" or "all".
							Default values is "no".
  
  -maskDir MASK_DIR, 		Path where background mask will be saved.
							Default values is 'none'.
  
  -mapK MAP_K_VALUES, 		Which values of k use to evaluate using MAP. '<k1,k2,k3,...,>' is the input format.
						    Default value is <1,5>.
  
  -gtR GT_RESULT, 			Ground-truth result of query.
							Default values is 'none'(no evaluating generated results).
  
  -gtM GT_MASKS, 			Path where ground-truth masks are.
							Default value is 'none' (no evaluating generated masks).
 ```
