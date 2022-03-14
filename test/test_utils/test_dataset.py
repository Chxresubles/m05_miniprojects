#!/usr/bin/env python

"""
Unit test for the handling of the datasets utility functions.

This file contains unit tests for the utility functions for
the handling of the datasets for the ML algorithms.

This file can be run using pytest with:

$ pytest test/test_utils/test_dataset.py -vv -s
"""


# ============================================================================================================
# Imports
# ============================================================================================================

import pytest
from utils import dataset


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================


# ============================================================================================================
# Unit tests
# ============================================================================================================

def test_get_red_train():
    """Check that the get function returns the correct red train dataset.
    """
    X, Y = dataset.get('red', 'train')

    assert X.shape[0] == Y.shape[0], 'Both inputs and outputs should have the same length'


def test_get_red_test():
    """Check that the get function returns the correct red train dataset.
    """
    X, Y = dataset.get('red', 'test')

    assert X.shape[0] == Y.shape[0], 'Both inputs and outputs should have the same length'


def test_get_white_train():
    """Check that the get function returns the correct red train dataset.
    """
    X, Y = dataset.get('white', 'train')

    assert X.shape[0] == Y.shape[0], 'Both inputs and outputs should have the same length'


def test_get_white_test():
    """Check that the get function returns the correct red train dataset.
    """
    X, Y = dataset.get('white', 'test')

    assert X.shape[0] == Y.shape[0], 'Both inputs and outputs should have the same length'


def test_get_house_train():
    """Check that the get function returns the correct red train dataset.
    """
    X, Y = dataset.get('houses', 'train')

    assert X.shape[0] == Y.shape[0], 'Both inputs and outputs should have the same length'


def test_get_house_test():
    """Check that the get function returns the correct red train dataset.
    """
    X, Y = dataset.get('houses', 'test')

    assert X.shape[0] == Y.shape[0], 'Both inputs and outputs should have the same length'


# ============================================================================================================
# Main function
# ============================================================================================================

if __name__ == '__main__':
    pytest.main()
