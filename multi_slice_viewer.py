# File: multi_slice_viewer.py
# Author: Brandon Curl
# Contact: brandoncurl@utexas.edu
# Adapted from https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
#
# Date Last Modified: 2-Jun-2021 09:30
# Description of Program: Configures a matlibplot.pyplot plot for 3D numpy array with scrolling to change slices

import matplotlib.pyplot as plt
import numpy as np

def multi_slice_viewer(volume):
    """Configures a plot for 3D numpy array with scrolling to change slices"""

    if type(volume) != np.ndarray:
        return f"Expected <class 'numpy.ndarray'> but got {type(volume)} instead"

    if np.squeeze(volume).ndim != 3:
        return f"Expected 3-dimensional squeezed array but got {np.squeeze(volume).ndim}-dimensional array instead"

    volume = np.rot90(np.squeeze(volume), k = 3, axes = (1, 2)) # Rotated around axes 1 and 2 three times for proper orientation
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2 # Viewing is initialized in the middle of the image depth
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    plt.show()


def mouse_click(event):
    """Clicking the mouse brings the figure to the middle of the image depth"""
    fig = event.canvas.figure
    ax = fig.axes[0]
    volume = ax.volume
    ax.index = volume.shape[0] // 2
    ax.images[0].set_array(volume[ax.index])
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
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0] # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def previous_slice(ax):
    """Go to the previous slice - triggered by scrolling down"""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
