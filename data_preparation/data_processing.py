import os
import glob
from tqdm import tqdm
import random

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
# From https://keras.io/examples/vision/3D_image_classification/
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    volume = volume - np.min(volume)
    volume = volume / (np.max(volume) - np.min(volume))
    volume = volume.astype('float32')
    return volume

def resize_volume(img, image_size, depth_size):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = depth_size
    desired_width = image_size
    desired_height = image_size
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path, image_size, depth_size):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume, image_size, depth_size)
    return volume

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-15,-10, -5, 5, 10,15]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

#build dataset for 3D model training
def build_dataset(scans, labels, train_index, val_index,\
                  batch_size_train, batch_size_val,\
                  epochs):
    x_train = np.array([scans[i] for i in train_index])
    y_train = np.array([labels[i] for i in train_index])
    x_val = np.array([scans[i] for i in val_index])
    y_val = np.array([labels[i] for i in val_index])
    
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(19)
        .map(train_preprocessing)
        .batch(batch_size_train, drop_remainder=True)
        .prefetch(AUTOTUNE)
        .repeat(epochs)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(19)
        .map(validation_preprocessing)
        .batch(batch_size_val, drop_remainder=True)
        .prefetch(AUTOTUNE)
        .repeat(epochs)
    )
    return train_dataset, validation_dataset