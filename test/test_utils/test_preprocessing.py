#!/usr/bin/env python

"""
Unit test for the preprocessing utility functions.

This file contains unit tests for the utility functions for
the preprocessing of the data for the ML algorithms.

This file can be run using pytest with:

$ pytest test/utils/test_dataset.py -vv -s
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

    assert np.array_equal(mean, np.array([0, 0.5, 1]))
    assert np.array_equal(std, np.array([1, 0.5, 0]))


def test_normalize():
    """Check that the normalize function works correctly.
    """
    data = np.array([[-1, 0, 1], [1, 1, 1]])
    mean = 0.5
    std = 0.5
    norm_data = preprocessing.normalize(data, [mean, std])

    assert data.shape == norm_data.shape
    assert norm_data.mean() == mean
    assert norm_data.std() == std


def test_min_max_scaling():
    """Check that the min max scaling works correctly.
    """


def test_z_norm():
    """Check that the z normalisation works correctly.
    """


def test_poly_2():
    """Check that the poly function generate degree 2 polynomials of the data.
    """


# ============================================================================================================
# Main function
# ============================================================================================================

if __name__ == '__main__':
    pytest.main()
