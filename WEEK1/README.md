# Week 1

Image retrieval using 1D histograms. If images contain background, it has to be removed.

## Usage:

### Descriptors generation:

``
$ generateDescriptors.py [-iDir input_dir] [-oDir output_dir] [-c color_space] [-maskDir mask_dir]
``

```

Options:

  -iDir, input_dir       path to the images folder.
  
  -oDir, output_dir      path to the folder where descriptor will be saved.
  
  -c, color_space		color space to use to compute descriptors: 'rgb', 'hsv', 'cielab', 'cieluv', 'ycbcr', 'all'. 
						If 'all' is given, descriptors of every color space will be generated.
                        
  -maskDir, mask_dir 	path to the masks of image backgrounds (OPTIONAL). 
  Default = None
 ```

### Compute retrieval



### Background removal


### Image retrieval evaluation




### Background removal evaluation

``
$ metricsEval.py [-aDir actual_input_dir] [-pDir predicted_input_dir]
``
```
Options:

  -aDir, actual_input_dir       path to the actual/ ground-truth background masks images folder.
  
  -pDir, predicted_input_dir      path to the predicted background mask image folder.
 
  Default = None

```