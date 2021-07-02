# File: DSC.py
# Author: Brandon Curl
# Contact: brandoncurl@utexas.edu
#
# Date Last Modified: 2-Jun-2021 09:45
# Description of Program: Calculates the Sorensen-Dice Coefficient of two 2D numpy masks

import numpy as np

def DSC(m1, m2):
    """Takes two arrays of data and returns the DICE coefficient"""

    if type(m1) != np.ndarray or type(m2) != np.ndarray:
        return f"Expected <class 'numpy.ndarray'> but got {type(m1)} and {type(m2)} instead"

    if m1.shape != m2.shape:
        return f"Arrays do not have equal shapes: {m1.shape} and {m2.shape}"

    addition_mask = m1 + m2
    true_positives = (addition_mask == 2).sum()
    false_positive_negative = (addition_mask == 1).sum()

    if 2 * true_positives + false_positive_negative == 0:
        return 'nan' # undefined DSC due to division by 0

    DSC = (2 * true_positives) / (2 * true_positives + false_positive_negative)

    return DSC