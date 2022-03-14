#!/usr/bin/env python

"""
Unit test for the preprocessing utility functions.

This file contains unit tests for the utility functions for
the preprocessing of the data for the ML algorithms.

This file can be run using pytest with:

$ pytest test/test_utils/test_preprocessing.py -vv -s
"""


# ============================================================================================================
# Imports
# ============================================================================================================

import numpy as np
import pytest
import preprocessing


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================


# ============================================================================================================
# unit tests
# ============================================================================================================

def test_estimate_norm():
    """Check that the estimate norm function returns the correct mean and standard deviation.
    """
    data = np.array([[-1, 0, 1], [1, 1, 1]])
    mean, std = preprocessing.estimate_norm(data)

    assert np.array_equal(mean, np.array([0, 0.5, 1])), 'Wrong estimation of the mean of the data'
    assert np.array_equal(std, np.array([1, 0.5, 0])), 'Wrong estimation of the standard deviation of the data'


def test_min_max_scaling_boundaries():
    """Check that the min max scaling fits values inside 0 and 1.
    """
    data = np.array([[-45, 25, 0.1638], [1.247, -421, 0]])
    scaled_data = preprocessing.min_max_scaling(data)

    assert np.all(scaled_data >= 0), 'Some values are lower than 0'
    assert np.all(scaled_data <= 1), 'Some values are higher than 1'


def test_min_max_scaling_values():
    """Check that the min max scaling fits values for each column.
    """
    data = np.array([[-1, 0, 1], [1, 1, 1]])
    scaled_data = preprocessing.min_max_scaling(data)

    assert np.array_equal(scaled_data, np.array([[0, 0, 0], [1, 1, 0]])), 'Wrong scaling of the data over [0, 1]'


def test_z_norm_values():
    """Check that the z normalisation computes the standard score of each value.
    """
    data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    score = np.array([[-1, -1], [-1, -1], [1, 1], [1, 1]])
    scaled_data = preprocessing.z_norm(data)

    assert np.array_equal(scaled_data, score), 'Wrong estimation of the standard score'


def test_poly_2():
    """Check that the poly function generate degree 2 polynomials of the data.
    """
    data = np.array([[0, 1], [2, 3], [4, 5]])
    polys = np.array([[1, 0, 1, 0, 0, 1], [1, 2, 3, 4, 6, 9], [1, 4, 5, 16, 20, 25]])

    poly_data = preprocessing.poly(data, 2)

    assert np.array_equal(poly_data, polys), 'Wrong computation of the polynomials'


# ============================================================================================================
# Main function
# ============================================================================================================

if __name__ == '__main__':
    pytest.main()
