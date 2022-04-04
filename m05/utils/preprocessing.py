#!/usr/bin/env python

"""
Preprocessing functions for the wine quality ML project.

This file was given with the course materials for the ex3
of the module M05 of the AI master at Idiap. Small
modifications were done to fit the file with our needs
for this mini-project.
"""


# ============================================================================================================
# Imports
# ============================================================================================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
import numpy as np


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================

def estimate_norm(data):
    """Estimates the mean and the std of the data

    Parameters:
        data : ndarray
            Numpy array that contains the data

    Returns:
        mean_and_std : (float, float)
            Mean and std of the data
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')

    return data.mean(axis=0), data.std(axis=0)


def min_max_scaling(data):
    """Min max scaling of the data

    Parameters:
        data : ndarray
            Numpy array that contains the data

    Returns:
        arr : ndarray
            Scaled array (value from 0 to 1)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')

    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def z_norm(data):
    """Z normalisation of the data

    Parameters:
        data : ndarray
            Numpy array that contains the data

    Returns:
        arr : ndarray
            Normalized array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')

    scaler = StandardScaler()
    return scaler.fit_transform(data)


def poly(data, degree):
    """Generate polynomial features from the data

    Parameters:
        data : ndarray
            Numpy array that contains the data
        degree : int
            The maximal degree of the polynomial features

    Returns:
        polys : ndarray
            New feature matrix consisting of all polynomial combinations
            of the features with degree less than or equal to the specified degree
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')
    if not isinstance(degree, int):
        raise TypeError('degree must be a int')

    poly_feats = PolynomialFeatures(degree)
    return poly_feats.fit_transform(data)
