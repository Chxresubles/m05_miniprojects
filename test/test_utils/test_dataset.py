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
import numpy as np
from m05.utils import dataset


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================


# ============================================================================================================
# Unit tests
# ============================================================================================================

def test_load_dataset_params_errors():
    """Check that the load_dataset function throws an error when a wrong parameter is given.
    """
    with pytest.raises(TypeError):
        dataset.load_dataset(0)
    with pytest.raises(ValueError):
        dataset.load_dataset('blueberries')


def test_load_params_errors():
    """Check that the load function throws an error when a wrong parameter is given.
    """
    with pytest.raises(TypeError):
        dataset.load(0, 'true')


@pytest.mark.parametrize("data, subset, splits", [(1, 'set1', {}),
                                                  (np.zeros((1, 1)), 1, {}),
                                                  (np.zeros((1, 1)), 'set1', 1)])
def test_split_data_type_errors(data, subset, splits):
    """Check that the split_data function throws a Type error when a wrong parameter is given.
    """
    with pytest.raises(TypeError):
        dataset.split_data(data, subset, splits)


def test_split_data_value_errors():
    """Check that the split_data function throws a Value error when a wrong parameter is given.
    """
    with pytest.raises(ValueError):
        dataset.split_data(np.zeros((1, 1)), 'indy_set', {})


@pytest.mark.parametrize("protocol, subset, part", [('red', 'set1', 'train'), ('red', 'set1', 'test'),
                                                    ('white', 'set1', 'train'), ('white', 'set1', 'test'),
                                                    ('houses', 'set1', 'train'), ('houses', 'set1', 'test'),

                                                    ('red', 'set2', 'train'), ('red', 'set2', 'test'),
                                                    ('white', 'set2', 'train'), ('white', 'set2', 'test'),
                                                    ('houses', 'set2', 'train'), ('houses', 'set2', 'test'),

                                                    ('red', 'set3', 'train'), ('red', 'set3', 'test'),
                                                    ('white', 'set3', 'train'), ('white', 'set3', 'test'),
                                                    ('houses', 'set3', 'train'), ('houses', 'set3', 'test')])
def test_get_right_params(protocol, subset, part):
    """Check that the get function returns a correct dataset.
    """
    X, Y = dataset.get(protocol, subset, part)

    assert X.shape[0] == Y.shape[0], 'Both inputs and outputs should have the same length'


@pytest.mark.parametrize("protocol, subset, part", [(1, 'set1', 'train'),
                                                    ('red', 1, 'train'),
                                                    ('red', 'set1', 1)])
def test_get_type_errors(protocol, subset, part):
    """Check that the get function throws a Type error when a wrong parameter is given.
    """
    with pytest.raises(TypeError):
        dataset.get(protocol, subset, part)


@pytest.mark.parametrize("protocol, subset, part", [('rainbow', 'set1', 'train'),
                                                    ('red', 'new_set', 'train'),
                                                    ('red', 'set1', 'indy_test')])
def test_get_value_errors(protocol, subset, part):
    """Check that the get function throws a Value error when a wrong parameter is given.
    """
    with pytest.raises(ValueError):
        dataset.get(protocol, subset, part)


# ============================================================================================================
# Main function
# ============================================================================================================

if __name__ == '__main__':
    pytest.main()
