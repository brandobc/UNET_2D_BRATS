# File: multi_slice_overlaion.py
# Author: Brandon Curl
# Contact: brandoncurl@utexas.edu
# Adapted from https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
#
# Date Last Modified: 7-Jun-2021 16:30
# Description of Program: Configures a matlibplot.pyplot plot for 3D numpy array with scrolling to change slices with a mask overlain

import matplotlib.pyplot as plt
import numpy as np

def multi_slice_overlain(volume1, volume2, alpha):
    """Configures a plot for 3D numpy array with scrolling to change slices with a mask overlain"""

    if type(volume1) != np.ndarray or type(volume2) != np.ndarray:
        return f"Expected <class 'numpy.ndarray'> but got {type(volume1)} and {type(volume2)} instead"

    if np.squeeze(volume1).ndim != 3 or np.squeeze(volume2).ndim != 3:
        return f"Expected 3-dimensional squeezed array but got {np.squeeze(volume1).ndim}-dimensional array and {np.squeeze(volume2).ndim}-dimensional array instead"

    volume1 = np.rot90(np.squeeze(volume1), k = 3, axes = (1, 2)) # Rotated around axes 1 and 2 three times for proper orientation
    volume2 = np.rot90(np.squeeze(volume2), k = 3, axes = (1, 2)) # Rotated around axes 1 and 2 three times for proper orientation
    fig, ax = plt.subplots()
    ax.volume1 = volume1
    ax.volume2 = volume2
    ax.index = volume1.shape[0] // 2 # Viewing is initialized in the middle of the image depth
    ax.imshow(volume1[ax.index])
    ax.imshow(volume2[ax.index], alpha = alpha)
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    plt.show()


def mouse_click(event):
    """Clicking the mouse brings the figure to the middle of the image depth"""
    fig = event.canvas.figure
    ax = fig.axes[0]
    volume1 = ax.volume1
    volume2 = ax.volume2
    ax.index = volume1.shape[0] // 2
    ax.images[0].set_array(volume1[ax.index])
    ax.images[1].set_array(volume2[ax.index])
    fig.canvas.draw()

def mouse_scroll(event):
    """Scrolling the mouse changes the viewing slice"""
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'down':
        previous_slice(ax)
    elif event.button == 'up':
        next_slice(ax)
    fig.canvas.draw()

def next_slice(ax):
    """Go to the next slice - triggered by scrolling up"""
    volume1 = ax.volume1
    volume2 = ax.volume2
    ax.index = (ax.index + 1) % volume1.shape[0] # wrap around using %
    ax.images[0].set_array(volume1[ax.index])
    ax.images[1].set_array(volume2[ax.index])

def previous_slice(ax):
    """Go to the previous slice - triggered by scrolling down"""
    volume1 = ax.volume1
    volume2 = ax.volume2
    ax.index = (ax.index - 1) % volume1.shape[0]  # wrap around using %
    ax.images[0].set_array(volume1[ax.index])
    ax.images[1].set_array(volume2[ax.index])