# Week 3

Image retrieval using multiresolution color 2D histograms, HoG texture descriptors and text descriptors. If images contain background, it has to be removed and if it contains text boxes, they have to be detected. In some images there are more than one painting. Some images contain noise.

## Usage:
``main.py`` can be used to process week 3 queries (``qsd1_w3`` and ``qsd2_w3``). 

### Example

Compute ``qsd2_w3`` query set, with multiresolution (3 levels) color, texture and text descriptors with weights [0.1667, 0.3333, 0.5] in the same order. Color and texture differences computed using L1 distance function, and the text descriptors differences computed using Cosine Simularity.

``
main.py -bbddDir ../../WEEK1/BBDD/ -qDir ../../WEEK3/qsd2_w3/ -noise yes -dType <color,texture,text> -wColor 0.1667 -wTexture 0.3333 -wText 0.5 -tBox yes -mPaintings yes -bRem yes -maskDir ./masks/ -gtR ../../WEEK3/qsd2_w3/gt_corresps.pkl -gtM ../../WEEK3/qsd2_w3/ -gtT ../../WEEK3/qsd2_w3/text_boxes.pkl
``

``
 main.py [-h] [-bbddDir BBDD_DIR] [-qDir QUERY_DIR]
               [-dDir DESCRIPTOR_DIR] [-dTrans TRANSCRIPTION_DIR] [-noise NOISE] [-dType DES_TYPE]
               [-wColor WEIGHT_COLOR] [-wTexture WEIGHT_TEXTURE]
               [-wText WEIGHT_TEXT] [-tBox TEXT_BOXES]
               [-tBoxDir TEXT_BOXES_DIR] [-rDir RESULTS_DIR] [-rK RESULT_K]
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
                        Indicate if there is noise in images. The default value is "no".
  -dType DES_TYPE, --des_type DES_TYPE
                        Indicate the descriptor combination. The default value is "<color,texture,text>".
  -wColor WEIGHT_COLOR, --weight_color WEIGHT_COLOR
                        Weight of the color descriptors in the combination. The default value is 1.
  -wTexture WEIGHT_TEXTURE, --weight_texture WEIGHT_TEXTURE
                        Weight of the texture descriptors in the combination. The default value is 1.
  -wText WEIGHT_TEXT, --weight_text WEIGHT_TEXT
                        Weight of the text descriptors in the combination. The default value is 1.
  -tBox TEXT_BOXES, --text_boxes TEXT_BOXES
                        Indicate if text boxes has to be detected: yes or no. The default value is no.
  -tBoxDir TEXT_BOXES_DIR, --text_boxes_dir TEXT_BOXES_DIR
                        Path where detected text boxes will be saved. The default value is ./textBoxes/.
  -rDir RESULTS_DIR, --results_dir RESULTS_DIR
                        Path where retrieval results will be saved. The default value is ./results/.
  -rK RESULT_K, --result_k RESULT_K
                        Number of predictions saved for each image in results. The default value is 10.
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
  -gtT GT_TEXT_BOXES, --gt_text_boxes GT_TEXT_BOXES
                        Path where ground-truth text boxes are. The default value is "None".
              
 ``
