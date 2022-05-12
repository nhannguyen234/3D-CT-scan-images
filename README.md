# 3D-scan-images
Here is the first version, it will be updated with more functions.

3D scan images is based on https://keras.io/examples/vision/3D_image_classification/

![image](https://user-images.githubusercontent.com/33461503/135021457-7d98830a-2dbe-4466-8a4d-e0d3b4bdda64.png)

USAGE:
- The format image should be dicom file. This version is not used for other formats as original images.
- 'training.py' is used to train a model. To modify the [image_size, depth_size], you need to locate to 'config.py' in data_preparation folder.
- Running 'training.py' will write weights to disk to a .h5 files. These files can then be loaded by 'testing.py' and the results will be saved in a csv format file.

      !python ./training.py --root_path 'your root path for training'\
                            --save_png 'your save png folder path for training'\
                            --save_nifti 'your save nifti folder path for training'\
                            --num_epochs 100\
                            --batch_size 32\
                            --output_weights_path 'your output weight path'
  
  + The 'training.py' will create '5 folds' model weights as default, if you want to modify the number of folds, you can locate to 'tf_cv_training.py' and change this in model folder. 
  + To run the training, the command should be:

  Note: 
  + root_path: your root path contains all folders with 'dicom' images. \
  + you can change num_epochs and batch_size if you want. \
  + all paths exclude filename and its extension, only path
             
- Evaluate your model by using 'testing.py' as the command below:

      !python ./testing.py --nifti_path 'your save nifti files path for testing'\
                           --input_weights_path 'your input weight path'
                           

                        
  ---> The results will be saved in './prediciton_file.csv'
