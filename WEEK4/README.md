# Week 4

Image retrieval using keypoint detection and local descriptors. If images contain background, it has to be removed and if it contains text boxes, they have to be detected. In some images there are more than one painting (maximum 3). Some images contain noise.

## Usage:
``main.py`` can be used to process week 4 query (``qsd1_w4``). 

### Example

Compute ``qsd1_w4`` query set, with ORB keypoints (max 2000) and descriptors. 
``
main.py -bbddDir ../../WEEK1/BBDD/ -qDir ../../WEEK4/qsd1_w4/ -dType new -ldType orb -gtR ../../WEEK4/qsd1_w4/gt_corresps.pkl -gtM ../../WEEK4/qsd1_w4/ -gtT ../../WEEK4/qsd1_w4/text_boxes.pkl
``

Compute ``qsd1_w4`` query set, with color, texture and text descriptors. 
``
main.py -bbddDir ../../WEEK1/BBDD/ -qDir ../../WEEK4/qsd1_w4/ -dType old -gtR ../../WEEK4/qsd1_w4/gt_corresps.pkl -gtM ../../WEEK4/qsd1_w4/ -gtT ../../WEEK4/qsd1_w4/text_boxes.pkl
``

``
 main.py [-h] [-bbddDir BBDD_DIR] [-qDir QUERY_DIR]
               [-dDir DESCRIPTOR_DIR] [-dTrans TRANSCRIPTION_DIR]
               [-noise NOISE] [-dType DES_TYPE] [-ldType LOCAL_DES_TYPE]
               [-tBox TEXT_BOXES] [-tBoxDir TEXT_BOXES_DIR]
               [-rDir RESULTS_DIR] [-rK RESULT_K]
               [-mPaintings MULTIPLE_PAINTINGS] [-bRem BACKGROUND_REM]
               [-maskDir MASK_DIR] [-mapK MAP_K_VALUES] [-gtR GT_RESULT]
               [-gtM GT_MASKS] [-gtT GT_TEXT_BOXES]
``

```

Options:

-h, --help            show this help message and exit
  -bbddDir BBDD_DIR, --BBDD_dir BBDD_DIR
                        Path of bbdd images.
  -qDir QUERY_DIR, --query_dir QUERY_DIR
                        Path of query images.
  -dDir DESCRIPTOR_DIR, --descriptor_dir DESCRIPTOR_DIR
                        Path where descriptors will be saved. The default value is ./descriptors/.
  -dTrans TRANSCRIPTION_DIR, --transcription_dir TRANSCRIPTION_DIR
                        Path where text transcriptions will be saved. The default value is ./textTranscriptions/.
  -noise NOISE, --noise NOISE
                        Indicate if there is noise in images. The default value is "yes".
  -dType DES_TYPE, --des_type DES_TYPE
                        Indicate to use descriptor the new (local descriptors)
                        or old (<color,texture,text>). The default values is new.
  -ldType LOCAL_DES_TYPE, --local_des_type LOCAL_DES_TYPE
                        Indicate which method for generating local
                        descriptors: sift, orb, harrisLaplace, brief or all. The default value is all.
  -tBox TEXT_BOXES, --text_boxes TEXT_BOXES
                        Indicate if text boxes has to be detected: yes or no. The default value is yes.
  -tBoxDir TEXT_BOXES_DIR, --text_boxes_dir TEXT_BOXES_DIR
                        Path where detected text boxes will be saved. The default value is ./textBoxes/.
  -rDir RESULTS_DIR, --results_dir RESULTS_DIR
                        Path where retrieval results will be saved. The default value is ./results/.
  -rK RESULT_K, --result_k RESULT_K
                        Number of predictions saved for each image in results. The default value is 10.
  -mPaintings MULTIPLE_PAINTINGS, --multiple_paintings MULTIPLE_PAINTINGS
                        Indicate if in the images could be multiple paintings: yes or no. The default value is yes.
  -bRem BACKGROUND_REM, --background_rem BACKGROUND_REM
                        Indicate if the query images have background: yes or no. The default value is yes.
  -maskDir MASK_DIR, --mask_dir MASK_DIR
                        Path where background mask will be saved. The default value is ./masks/.
  -mapK MAP_K_VALUES, --map_k_values MAP_K_VALUES
                        Which values of k use to evaluate using MAP. The default value is <1,5>.
  -gtR GT_RESULT, --gt_result GT_RESULT
                        Ground-truth result of query. The default value is "None".
  -gtM GT_MASKS, --gt_masks GT_MASKS
                        Path where ground-truth masks are. The default value is "None".
  -gtT GT_TEXT_BOXES, --gt_text_boxes GT_TEXT_BOXES
                        Path where ground-truth text boxes are. The default value is "None".
              
 ``
