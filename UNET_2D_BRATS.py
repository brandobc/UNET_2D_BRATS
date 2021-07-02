# File: UNET_2D_BRATS.py
# Author: Brandon Curl
# Contact: brandoncurl@utexas.edu
#
# Date Last Modified: 2-Jun-2021 12:00
# Description of Program: Creates and trains a 2D UNET on the 2020 BRATS Dataset

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np
from load_BRATS_data import *

IMG_HEIGHT = 240
IMG_WIDTH = 240
IMG_DEPTH = 1

# Imports the BRATS data to numpy arrays
X_train, Y_train, X_test, Y_test, train_IDs, test_IDs = load_BRATS_data(
    volumes = 369,
    slices = 155,
    img_height = IMG_HEIGHT,
    img_width = IMG_WIDTH,
    img_depth = IMG_DEPTH,
    path = 'C:/Users/BJCurl/PycharmProjects/BRATS_2020_data')


def get_model(width = IMG_WIDTH, height = IMG_HEIGHT, depth = IMG_DEPTH):
    """Build a 2D UNET"""

    inputs = layers.Input((width, height, depth))

    # Contraction path
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Define the model
    model = Model(inputs, outputs, name = "UNET_2D_BRATS")
    return model

# Build the model
model = get_model()
model.summary()

# Compile and train model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')

callbacks = [
    callbacks.EarlyStopping(patience = 3, monitor = 'val_loss'),
    callbacks.TensorBoard(log_dir = 'logs'),
    callbacks.ModelCheckpoint('UNET_2D_BRATS_training_checkpoint', verbose = 1, save_best_only = True)
]

model.fit(X_train, Y_train, validation_split = 0.1, batch_size = 16, epochs = 25, callbacks = callbacks)
model.save('UNET_2D_BRATS_trained_model')

# Evaluate the model on test data
model.evaluate(X_test, Y_test)