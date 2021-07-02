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

#TRAIN_IDs = {1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 31, 32, 33, 34, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 186, 187, 188, 190, 192, 193, 194, 195, 196, 197, 198, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 238, 239, 240, 241, 242, 244, 245, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 257, 258, 259, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 367, 369}
TEST_IDs = {4, 133, 6, 13, 22, 24, 29, 30, 35, 37, 41, 45, 49, 185, 59, 61, 189, 63, 191, 66, 199, 200, 205, 77, 89, 346, 222, 96, 226, 237, 366, 368, 243, 119, 253, 126, 127}

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