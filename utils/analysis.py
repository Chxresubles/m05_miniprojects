#!/usr/bin/env python

"""
Utility functions for the result analysis.

This file contains utility functions for
the calculations of metrics for the
ML algorithms need for result analysis.
"""


# ============================================================================================================
# Imports
# ============================================================================================================

import numpy as np


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================

def MAE(prediction, ground_truth):
    """Calculate the mean absolute error for the predictions compared to the ground truth."""
    if not isinstance(prediction, np.ndarray):
        raise TypeError('prediction must be a numpy array')
    if not isinstance(ground_truth, np.ndarray):
        raise TypeError('ground_truth must be a numpy array')
    if prediction.shape == ground_truth.shape:
        raise TypeError('prediction and ground_truth must have the same shape')
    if len(prediction.shape) != 1:
        raise TypeError('prediction and ground_truth must be Nx1 vectors')
    
    errors = np.absolute(np.subtract(prediction, ground_truth)).sum()
    return (errors / len(prediction))
