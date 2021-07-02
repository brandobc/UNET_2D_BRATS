import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np
from load_BRATS_data import *
import matplotlib.pyplot as plt
import random

IMG_WIDTH = 240
IMG_HEIGHT = 240
IMG_DEPTH = 1

# Imports the BRATS data to numpy arrays
X_train, Y_train, X_test, Y_test, train_IDs, test_IDs = load_BRATS_data(
    volumes = 5,
    slices = 155,
    img_width = IMG_WIDTH,
    img_height = IMG_HEIGHT,
    img_depth = IMG_DEPTH,
    path = 'C:/Users/Brandon/OneDrive/Documents/College Things/MD Anderson/Chung Lab/BRATS-2020-Data',
    train_IDs = set([5])
)

i = 90

x = X_train[i, :, :, 0]
y = Y_train[i, :, :, 0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(x)
ax2.imshow(y)
ax3.imshow(x)
ax3.imshow(y, alpha = 0.5)
plt.show()
