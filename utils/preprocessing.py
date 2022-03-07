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

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================

def estimate_norm(data):
    """Estimates the mean and the std of the data

    Args:
        data (ndarray): array that contains the data

    Returns:
        (float,float): mean and std of the data
    """
    return data.mean(axis=0), data.std(axis=0, ddof=1)


def normalize(data, norm):
    """Normalize the data using the given mean and std

    Args:
        data (ndarray): array that contains the data
        norm (tuple): tuple containing the mean and std

    Returns:
        ndarray: normalized array
    """
    return np.array([(k - norm[0]) / norm[1] for k in data])


def min_max_scaling(data):
    """Min max scaling of the data

    Args:
        data (ndarray): array that contains the data

    Returns:
        ndarray: scaled array (value from 0 to 1)
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def z_norm(data):
    """Z normalisation of the data

    Args:
        data (ndarray): array that contains the data

    Returns:
        ndarray: normalized array
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def poly(data, degree):
    """Generate polynomial features from the data

    Args:
        data (ndarray): array that contains the data
        degree (int): the maximal degree of the polynomial features

    Returns:
        ndarray: new feature matrix consisting of all polynomial combinations
        of the features with degree less than or equal to the specified degree
    """
    poly_feats = PolynomialFeatures(degree)
    return poly_feats.fit_transform(data)
