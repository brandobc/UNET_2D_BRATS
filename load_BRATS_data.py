# File: load_BRATS_data.py
# Author: Brandon Curl
# Contact: brandoncurl@utexas.edu
#
# Date Last Modified: 2-Jun-2021 11:00
# Description of Program: Creates and returns 4D numpy arrays that contain the 2020 BRATS T2 FLAIR Images and Hyperintensity Masks

import numpy as np
import random
from skimage.transform import resize
import h5py


def load_BRATS_data(volumes, slices, img_width, img_height, img_depth, path, train_IDs = None, test_IDs = None):
    """Creates and returns 4D numpy arrays that contain the 2020 BRATS T2 FLAIR Images and Hyperintensity Masks"""

    if train_IDs == None and test_IDs == None: # Randomly parses the data by patient between the training and testing sets

        train_cut = round(volumes * 0.9)
        test_cut = round(volumes * 0.1)

        volume_IDs_all = set(range(1, volumes + 1))
        LGG = set(range(260, 335 + 1)) # Removes low-grade glioma patients
        volume_IDs = volume_IDs_all - LGG
        test_IDs = set()

        while len(test_IDs) < test_cut:
            test_IDs.add(random.randint(1, volumes))

        train_IDs = volume_IDs - test_IDs

    elif train_IDs == None:
        train_cut = 0
        test_cut = len(test_IDs)
        train_IDs = set()

    elif test_IDs == None:
        test_cut = 0
        train_cut = len(train_IDs)
        test_IDs = set()

    else:
        train_cut = len(train_IDs)
        test_cut = len(test_IDs)

    # Creates blank arrays to hold the data
    X_train = np.zeros((train_cut * slices, img_width, img_height, img_depth), dtype = np.float32)
    Y_train = np.zeros((train_cut * slices, img_width, img_height, img_depth), dtype = np.bool)

    X_test = np.zeros((test_cut * slices, img_width, img_height, img_depth), dtype = np.float32)
    Y_test = np.zeros((test_cut * slices, img_width, img_height, img_depth), dtype = np.bool)


    # Populates the arrays
    X_train, Y_train = populate_data(X = X_train, Y = Y_train, IDs = train_IDs, slices = slices, path = path, img_width = img_width, img_height = img_height)
    X_test, Y_test = populate_data(X = X_test, Y = Y_test, IDs = test_IDs, slices = slices, path = path, img_width = img_width, img_height = img_height)

    print("Training IDs:", train_IDs)
    print("Testing IDs:", test_IDs)
    return X_train, Y_train, X_test, Y_test, train_IDs, test_IDs


def populate_data(X, Y, IDs, slices, path, img_width, img_height):
    """Populates the data numpy arrays with the training and testing data"""

    i = 0

    for id in IDs:
        for z in range(slices):
            slice = h5py.File(((path + '/') if (path[-1] != '/') else path) + f'volume_{id}_slice_{z}.h5', 'r')
            image = slice['image']  # Array of 4 images corresponding to the key below
            mask = slice['mask']  # Array of 3 masks corresponding to the key below

            FLAIR = image[:, :, 0]  # already normalized with Gaussian distribution and set to (240, 240) slice dimensions
            # 0 is FLAIR
            # 1 is T1?
            # 2 is T1Gd
            # 3 is T2
            FLAIR = np.expand_dims(resize(FLAIR, (img_width, img_height), mode = 'constant', preserve_range = True), axis = -1)
            X[i] = FLAIR

            T2_hyperintensity = np.maximum(np.maximum(mask[:, :, 0], mask[:, :, 1]), mask[:, :, 2])  # enhancing disease, necrotic interior tissue, and peritumoral edema
            # 0 is necrotic
            # 1 is peritumoral edema
            # 2 is enhancing
            T2_hyperintensity = np.expand_dims(resize(T2_hyperintensity, (img_width, img_height), mode = 'constant', preserve_range = True), axis = -1)
            Y[i] = T2_hyperintensity

            slice.close()
            i += 1
    
    return X, Y
