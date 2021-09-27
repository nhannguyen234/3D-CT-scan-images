import os
import glob
import tqdm 
import pandas as pd
import numpy as np
from data_preparation.data_processing import process_scan

import tensorflow as tf
from tensorflow.keras.models import load_model

from optparse import OptionParser

AUTOTUNE = tf.data.experimental.AUTOTUNE

parser = OptionParser()

parser.add_option("--nifti_path", dest="nifti_path", help="Root path stores nifti file data.")
parser.add_option("--input_weights_path", dest="input_weights_path", help="Input path for weight.")

(options, args) = parser.parse_args()

path_nifti = options.nifti_path
input_weights_path = options.input_weights_path

def test_preprocessing(volume):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume

def get_test_dataset(x_test):
    dataset = (
        tf.data.Dataset.from_tensor_slices((x_test)).shuffle(len(x_test))
        .map(test_preprocessing)
        .batch(1)
        .prefetch(AUTOTUNE)
    )
    return dataset

#Load model weights
model_weights = [x for x in glob.glob(os.path.join(input_weights_path,'*.h5'))]
models = [load_model(model) for model in model_weights]

def predicting(test_dataset, models, name):
    weights ={
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 3}

    weights_sum = sum(weights.values())
    weights = {k: v/weights_sum for k, v in weights.items()}
    
    predict = [model.predict(test_dataset, verbose=1) for model in models]
    print(f'finished {name}')
    
    for i, pred in enumerate(predict):
        predict[i] = weights[i]*pred
    
    pred = sum(predict)
    del test_dataset #to release ram
    return pred

x_test = np.array([process_scan(path) for path in glob.glob(os.path.join(path_nifti,'*.nii'))])
test_dataset = get_test_dataset(x_test)
pred = predicting(test_dataset, models, '0')

#Export prediction in a csv file
cases = [os.path.basename(path).split('_')[0] for path in glob.glob(os.path.join(path_nifti,'*.nii'))]
df_final = pd.DataFrame(dict(zip(cases, pred)), columns=['cases','class'])
df_final.to_csv('./prediction_file.csv')