import os
import glob
import tqdm 
import pandas as pd
import numpy as np

import data_preparation
from data_preparation.data_format_converter import dcm2png, png2nifti
from data_preparation.data_processing import process_scan
from model.tf_cv_training import training
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--root_path", dest="root_path", help="Root path stores dicom file data.")
parser.add_option("--save_png", dest="save_png_path", help="Path to save image png format files.")
parser.add_option("--save_nifti", dest="save_nifti_path", help="Path to save nifti files.")
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=100)
parser.add_option("--batch_size", type='int', dest='batch_size', default=32)
parser.add_option("--output_weights_path", dest='output_weights_path', help="Output path for weights.", default='./')

(options, args) = parser.parse_args()

C = data_preparation.config.Config()

image_size = C.im_size
C.epochs = options.num_epochs
C.batch_size = options.batch_size

path_saved_png = options.save_png_path
path_saved_nifti = options.save_nifti_path
root_path = options.root_path

#Convert dicom files to png files
subdirs = [os.listdir(x) for x in glob.glob(root_path)]
for subdir in tqdm(subdirs[0], desc='Progress dcm2png'):
    folders_name = [os.listdir(x) for x in glob.glob(os.path.join(root_path,subdir))]
    for folder_name in folders_name[0]:
        path = os.path.join(root_path, subdir, folder_name,'*.dcm')
        dcm2png(path, path_saved_png, image_size, folder_name)

#Conver png files to nifti files
for subdir in tqdm(subdirs[0], desc='Progress png2nifti'):
    folders_name = [os.listdir(x) for x in glob.glob(os.path.join(root_path,subdir))]
    for folder_name in folders_name[0]:
        png2nifti(path_saved_png, path_saved_nifti, folder_name)

csv_path = './train_labels.csv'
df_label = pd.read_csv(csv_path)
cases = [x for x in df_label['case']]
dict_label = dict(zip(cases, df_label['class']))

scan_paths = glob.glob(os.path.join(path_saved_nifti,'*.nii'))
scans = np.array([process_scan(path) for path in scan_paths])
labels = np.array([dict_label[os.path.basename(path).split('_')[0]] for path in scan_paths])

# TRAINING MODEL

training(scans, labels, '0', C.epochs, C.batch_size)
