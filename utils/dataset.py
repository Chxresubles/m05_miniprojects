#!/usr/bin/env python

"""
Utility functions for the handling of the datasets.

This file was given with the course materials for the ex3
of the module M05 of the AI master at Idiap. Small
modifications were done to fit the file with our needs
for this mini-project.
"""


# ============================================================================================================
# Imports
# ============================================================================================================

import os
import numpy as np
import csv
import re


# ============================================================================================================
# Constants
# ============================================================================================================
PROTOCOLS = {
    'red': {'dataset': 'red wine', 'train': range(0, 1599, 2), 'test': range(1, 1599, 2)},
    'white': {'dataset': 'white wine', 'train': range(0, 4898, 2), 'test': range(1, 4898, 2)},
    'houses': {'dataset': 'houses', 'train': range(0, 506, 2), 'test': range(1, 506, 2)},
}

DATASETS = [
    'red wine',
    'white wine',
    'houses'
]

SUBSETS = [
    'train',
    'test',
]

WINE_VARIABLES = [
    'Fixed Acidity',
    'Volatile Acidity',
    'Citric Acid',
    'Residual Sugar',
    'Chlorides',
    'Free Sulfur dioxide',
    'Total Sulfur dioxide',
    'Density',
    'pH',
    'Sulphates',
    'Alcohol',
    'Quality'
]

HOUSE_VARIABLES = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PIRATIO',
    'B',
    'LSTAT',
    'MEDV',
]

WINE_FILE_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'wine_quality', 'dataset')

HOUSE_FILE_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'boston_house_prices', 'dataset')


# ============================================================================================================
# Useful functions
# ============================================================================================================


def load_dataset(dataset):
    """Load the wanted dataset in a numpy array

    Args:
        dataset (string): name of the wanted dataset

    Returns:
        array: contains the wanted dataset
    """
    if not isinstance(dataset, str):
        raise TypeError('dataset must be a string')
    if dataset not in DATASETS:
        raise ValueError(f'dataset was not found in available datasets: {dataset}')

    data = []
    if dataset == 'red wine':
        with open(os.path.join(WINE_FILE_PATH, 'winequality-red.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=';')
            data = load(reader, True)
    elif dataset == 'white wine':
        with open(os.path.join(WINE_FILE_PATH, 'winequality-white.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=';')
            data = load(reader, True)
    elif dataset == 'houses':
        with open(os.path.join(HOUSE_FILE_PATH, 'housing.data'), 'rt') as f:
            with open(os.path.join(HOUSE_FILE_PATH, 'housing.csv'), 'wt') as fcsv:
                for line in f:
                    a_str = re.sub(' +', ';', line)
                    a_str = re.sub(' \n', '\n', a_str)
                    a_str.lstrip()
                    fcsv.write(a_str[1:])
        with open(os.path.join(HOUSE_FILE_PATH, 'housing.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=';')
            data = load(reader, False)
    return data


def load(reader, skip_first_line):
    """Load the content of the reader in a numpy array

    Args:
        reader (ReaderObject): Read a file data
        skip_first_line (boolean): skip first line or not

    Returns:
        ndarray: array that contains the data
    """
    if not isinstance(skip_first_line, bool):
        raise TypeError('skip_first_line must be a bool')

    data = []
    for k, row in enumerate(reader):
        if not k and skip_first_line:
            continue
        data.append(np.array([float(z) for z in row]))
    data = np.vstack(data)
    return data


def split_data(data, subset, splits):
    """Get the wanted subset from the data as numpy array

    Args:
        data (ndarray): contains the data
        subset (string): the wanted subset (either 'train' or 'test')
        splits (ndarray): list of index of the subset

    Returns:
        ndarray: array splited
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')
    if not isinstance(subset, str):
        raise TypeError('subset must be a str')
    if not isinstance(splits, dict):
        raise TypeError('splits must be a dict')
    if subset not in SUBSETS:
        raise ValueError(f'subset was not found in available subsets: {subset}')

    return data[splits[subset]]


def get(protocol, subset):
    """Retrieve the wanted data subset as two numpy arrays X and Y

    Args:
        protocol (string): which protocol we want to use (red, white, houses)
        subset (string): the wanted subset (either 'train' or 'test')

    Returns:
        (ndarray,ndarray): two array X and Y
    """
    if not isinstance(protocol, str):
        raise TypeError('protocol must be a str')
    if not isinstance(subset, str):
        raise TypeError('subset must be a str')
    if protocol not in PROTOCOLS:
        raise ValueError(f'protocol was not found in available protocols: {protocol}')
    if subset not in SUBSETS:
        raise ValueError(f'subset was not found in available subsets: {subset}')

    fullData = split_data(load_dataset(
        PROTOCOLS[protocol]['dataset']), subset, PROTOCOLS[protocol])
    return fullData.T[:fullData.shape[1]-1].T, fullData.T[-1].T.reshape(-1, 1)
