#!/usr/bin/env python

"""
Unit test for the result analysis utility functions.

This file contains unit tests for the utility functions for
the calculations of metrics for the ML algorithms needed
for result analysis.

This file can be run using pytest with:

$ pytest test/test_utils/test_analysis.py -vv -s
"""


# ============================================================================================================
# Imports
# ============================================================================================================

import numpy as np
import pytest
from utils import analysis


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================


# ============================================================================================================
# Unit tests
# ============================================================================================================

def test_MAE_noErrors():
    """Check that the mean absolute error function works properly with 0 errors.
    """
    pred = np.array([[1, 1]]).T
    gt = np.array([[1, 1]]).T

    assert analysis.MAE(pred, gt) == 0, 'Mean absolute error should be 0, (|1-1| + |1-1|)/2.'


def test_MAE_positiveError():
    """Check that the mean absolute error function works properly with a positive error.
    """
    pred = np.array([[1, 1]]).T
    gt = np.array([[1, 0]]).T

    assert analysis.MAE(pred, gt) == 0.5, 'Mean absolute error should be 0.5, (|1-1| + |1-0|)/2.'


def test_MAE_negativeErrors():
    """Check that the mean absolute error function works properly with a negative error.
    """
    pred = np.array([[1, 0]]).T
    gt = np.array([[1, 1]]).T

    assert analysis.MAE(pred, gt) == 0.5, 'Mean absolute error should be 0.5, (|1-1| + |0-1|)/2.'


def test_MAE_allErrors():
    """Check that the mean absolute error function works properly with only errors.
    """
    pred = np.array([[0, 0]]).T
    gt = np.array([[1, 1]]).T

    assert analysis.MAE(pred, gt) == 1, 'Mean absolute error should be 1, (|0-1| + |0-1|)/2.'


# ============================================================================================================
# Main function
# ============================================================================================================

if __name__ == '__main__':
    pytest.main()
