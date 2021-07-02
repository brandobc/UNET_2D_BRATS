import tensorflow as tf
from tensorflow.keras import layers, Model, models
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import h5py
from load_BRATS_data import *
from DSC import *

# Imports the BRATS data to numpy arrays
X_train, Y_train, X_test, Y_test = load_BRATS_data(VOLUMES = 2, SLICES = 155, IMG_WIDTH = 240, IMG_HEIGHT = 240, IMG_CHANNELS = 1)

# Loads the trained model
unet = models.load_model('BRATS_trained_model')

# Predict the test cases
preds = unet.predict(X_test, verbose = 1)

max_DSC_avg = 0.5
best_confidence = 100


for conf in np.arange(0.2, 1, 0.001):
    conf = float(conf)
    DSC_scores = []
    for i in range(len(preds)):
        pred_1_0 = preds[i, :, :, 0]
        pred_1_0[pred_1_0 < conf] = 0
        pred_1_0[pred_1_0 >= conf] = 1
        pred_1_0 = pred_1_0.astype(np.uint)
        y = Y_test[i, :, :, 0]
        y = y.astype(np.uint)
        DSC_scores.append(dice(pred_1_0, y))

    DSC_scores = [x for x in DSC_scores if str(x) != 'nan']

    if np.mean(DSC_scores) > max_DSC_avg:
        max_DSC_avg = np.mean(DSC_scores)
        best_confidence = conf