import os
import glob

import cv2 as cv
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import PIL
from PIL import Image

import SimpleITK as sitk
import nibabel as nib

from skimage.feature import hog
from skimage import exposure

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

def convert_data(img_data):
    img_data = img_data - np.min(img_data)
    img_data = img_data / (np.max(img_data) - np.min(img_data))
    img_data = (img_data * 255).astype(np.uint8)
    return img_data

def read_xray (path, voi_lut = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        img_data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        img_data = dicom.pixel_array
               
    return convert_data(img_data)

# ---- Crop each image to remove black background ----

def check_zero_area(image_data):
    zero_area = len(np.where(image_data==0)[0])
    total_area = len(image_data[0])*len(image_data[1])
    return zero_area/total_area

def non_zero_bbox_area(image_data):
    non_zero_area = np.where(image_data!=0)
    
    x_min = np.min(non_zero_area[0])
    x_max = np.max(non_zero_area[0])
    y_min = np.min(non_zero_area[1])
    y_max = np.max(non_zero_area[1])
    
    return x_min, x_max, y_min, y_max, (x_max-x_min)*(y_max-y_min)

def bbox_max_area(files_path, image_size): 
    list_bbox_area = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    for file in files_path:
        image_data = read_xray(file)
        image_data = resize(image_data, image_size)
        image_data = np.asarray(image_data)
        try: 
            x_min, x_max, y_min, y_max, area = non_zero_bbox_area(image_data)
            list_bbox_area.append(area)
            xmin.append(x_min)
            xmax.append(x_max)
            ymin.append(y_min)
            ymax.append(y_max)
        except ValueError: continue
    i = list_bbox_area.index(np.max(list_bbox_area)) #index of max value in list_bbox_area
    return xmin[i], xmax[i], ymin[i], ymax[i]


def dcm2png(path, saved_path, image_size, folder_name):
    files_path = [x for x in glob.glob(os.path.join(path,'*.dcm'))]
    try: x_min, x_max, y_min, y_max = bbox_max_area(files_path)
    except ValueError: 
        print(f'This {path} is failed')
    for file in files_path:
        image_data = read_xray(file)
        image_data = resize(image_data, image_size)
        image_data = np.asarray(image_data)
        file_name = os.path.basename(file).split('.')[0]
        if check_zero_area(image_data) < 0.95: #choose percent to eliminate unneccessary dataset
            image_data = image_data[x_min:x_max, y_min:y_max]
            # fd, hog_image = hog(image_data, orientations=8, pixels_per_cell=(3,3),
            #     cells_per_block=(1, 1), visualize=True)
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 255)).astype(np.uint8)
            cv.imwrite(f'{saved_path}/{folder_name}_{file_name}.png', image_data)

def png2nifti(path, saved_path, folder_name):
    file_names = glob.glob(f'{path}/{folder_name}_*.png')
    reader = sitk.ImageSeriesReader()
    try: 
        reader.SetFileNames(file_names)
        vol = reader.Execute()
        sitk.WriteImage(vol, f'{saved_path}/{folder_name}.nii')
    except RuntimeError: 
        print(f'This folder {folder_name} is failed')