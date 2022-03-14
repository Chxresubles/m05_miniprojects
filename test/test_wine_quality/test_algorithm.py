#!/usr/bin/env python

"""
Unit test for the algorithm functions for the wine quality.

This file contains unit tests for the ML algorithms
of the wine quality project.

This file can be run using pytest with:

$ pytest test/test_wine_quality/test_algorithm.py -vv -s
"""


# ============================================================================================================
# Imports
# ============================================================================================================

import numpy as np
import pytest
from wine_quality import algorithm


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================


# ============================================================================================================
# Unit tests
# ============================================================================================================

def test_LR_train():
    """Check that the Linear Regression training method works correctly.
    """
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y = np.dot(X, np.array([1, 2])) + 3

    model = algorithm.LR_train(X, Y)

    assert model.score == 0, 'Model score should be equal to one.'
    assert np.array_equal(model.coef_, np.array([1, 2])), 'Model coefficients should be equal to [1, 2].'


def test_LR_evaluate():
    """Check that the Linear Regression training method works correctly.
    """
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y = np.dot(X, np.array([1, 2])) + 3

    model = algorithm.LR_train(X, Y)
    algorithm.LR_evaluate(model, X, Y)


# ============================================================================================================
# Main function
# ============================================================================================================

if __name__ == '__main__':
    pytest.main()
