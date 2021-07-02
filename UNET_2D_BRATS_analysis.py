# File: UNET_2D_BRATS_analysis.py
# Author: Brandon Curl
# Contact: brandoncurl@utexas.edu
#
# Date Last Modified: 2-Jun-2021 13:00
# Description of Program: [INSERT]

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from load_BRATS_data import *
from DSC import *

#TRAIN_IDs = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 227, 228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259}
TEST_IDs = {2, 262, 264, 141, 270, 273, 146, 145, 283}#, 285, 30, 287, 32, 161, 35, 163, 186, 187, 64, 66, 75, 212, 89, 93, 95, 96, 226, 231, 109}

# Imports the BRATS data to numpy arrays
X_train, Y_train, X_test, Y_test, tr, te = load_BRATS_data(
    volumes = None,
    slices = 155,
    img_width = 240,
    img_height = 240,
    img_depth = 1,
    path = 'C:/Users/BJCurl/PycharmProjects/BRATS_2020_data',
    #train_IDs = TRAIN_IDs,
    test_IDs = TEST_IDs
)

# Loads the trained model
model = models.load_model('C:/Users/BJCurl/PycharmProjects/UNET_2D_BRATS/UNET_2D_BRATS_trained_model')

# Predict the test cases
preds = model.predict(X_test, verbose = 1)


max_DSC_avg = 0
best_confidence = 0

i = 75

for conf in np.arange(0.001, 1.002, 0.001):
    conf = float(conf)
    DSC_scores = []
    for i in range(len(preds)):
        pred_1_0 = np.copy(preds[i, :, :, 0])
        pred_1_0[pred_1_0 < conf] = 0
        pred_1_0[pred_1_0 >= conf] = 1
        pred_1_0 = pred_1_0.astype(np.uint)
        y = Y_test[i, :, :, 0]
        y = y.astype(np.uint)
        DSC_scores.append(DSC(pred_1_0, y))


    DSC_scores = [x for x in DSC_scores if str(x) != 'nan']

    if np.mean(DSC_scores) > max_DSC_avg:
        max_DSC_avg = np.mean(DSC_scores)
        best_confidence = conf