# File: UNET_2D_BRATS_analysis.py
# Author: Brandon Curl
# Contact: brandoncurl@utexas.edu
#
# Date Last Modified: 2-Jun-2021 13:00
# Description of Program: [INSERT]

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from load_BRATS_data import *
from DSC import *

TRAIN_IDs = set()
TEST_IDs = set()

# Imports the BRATS data to numpy arrays
X_train, Y_train, X_test, Y_test = load_BRATS_data(
    volumes = None,
    slices = 155,
    img_width = 240,
    img_height = 240,
    img_depth = 1,
    path = 'C:\Users\BJCurl\PycharmProjects\BRATS_2020_data',
    train_IDs = TRAIN_IDs,
    test_IDs = TEST_IDs
)[0:3]

# Loads the trained model
model = models.load_model('C:\Users\BJCurl\PycharmProjects\UNET_2D_BRATS\UNET_2D_BRATS_trained_model')

# Predict the test cases
preds = model.predict(X_test, verbose = 1)

max_DSC_avg = 0
best_confidence = 0


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