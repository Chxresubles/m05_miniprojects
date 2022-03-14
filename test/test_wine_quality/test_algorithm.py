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
    """Check that the Linear Regression training function works correctly with example sets.
    """
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y = np.dot(X, np.array([1, 2])) + 3

    model = algorithm.LR_train(X, Y)

    assert model.score(X, Y) == 1, 'Model score should be equal to one.'
    assert np.allclose(model.coef_, np.array([1, 2])), 'Model coefficients should be equal to [1, 2].'


def test_LR_evaluate():
    """Check that the Linear Regression evaluation function does not crash.
    """
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y = np.dot(X, np.array([1, 2])) + 3

    model = algorithm.LR_train(X, Y)
    algorithm.LR_evaluate(model, X, Y)


def test_RT_train():
    """Check that the Regression Tree training function works correctly with example sets.
    """
    X = np.array([[0, 0], [1, 1]])
    Y = np.array([[0, 1]]).T

    model = algorithm.RT_train(X, Y)

    assert model.score(X, Y) == 1, 'Model score should be equal to one.'


def test_RT_evaluate():
    """Check that the Regression Tree evaluation function does not crash.
    """
    X = np.array([[0, 0], [1, 1]])
    Y = np.array([[0, 1]]).T

    model = algorithm.RT_train(X, Y)
    algorithm.RT_evaluate(model, X, Y)


def test_trainAndTest_model():
    """Check that the trainAndTest function works with both red and white wine.
    """
    algorithm.trainAndTest('LR', 'minmax', 'train', 'red', 1)
    algorithm.trainAndTest('RT', 'minmax', 'train', 'red', 1)
    with pytest.raises(ValueError):
        algorithm.trainAndTest('random_model', 'minmax', 'train', 'red', 1)


def test_trainAndTest_norm():
    """Check that the trainAndTest function works with both red and white wine.
    """
    algorithm.trainAndTest('LR', 'minmax', 'train', 'red', 1)
    algorithm.trainAndTest('LR', 'znorm', 'train', 'red', 1)
    with pytest.raises(ValueError):
        algorithm.trainAndTest('LR', 'true_norm', 'train', 'red', 1)


def test_trainAndTest_evalSet():
    """Check that the trainAndTest function works with both red and white wine.
    """
    algorithm.trainAndTest('LR', 'minmax', 'train', 'red', 1)
    algorithm.trainAndTest('LR', 'minmax', 'test', 'red', 1)
    with pytest.raises(ValueError):
        algorithm.trainAndTest('LR', 'minmax', 'random_set', 'red', 1)


def test_trainAndTest_color():
    """Check that the trainAndTest function works with both red and white wine.
    """
    algorithm.trainAndTest('LR', 'minmax', 'train', 'red', 1)
    algorithm.trainAndTest('LR', 'minmax', 'train', 'white', 1)
    with pytest.raises(ValueError):
        algorithm.trainAndTest('LR', 'minmax', 'champagne', 'red', 1)


def test_trainAndTest_poly():
    """Check that the trainAndTest function works with both red and white wine.
    """
    algorithm.trainAndTest('LR', 'minmax', 'train', 'red', 0)
    algorithm.trainAndTest('LR', 'minmax', 'train', 'red', 1)
    algorithm.trainAndTest('LR', 'minmax', 'train', 'red', 2)


# ============================================================================================================
# Main function
# ============================================================================================================

if __name__ == '__main__':
    pytest.main()
