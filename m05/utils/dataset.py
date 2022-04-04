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
from sklearn.model_selection import train_test_split


# ============================================================================================================
# Constants
# ============================================================================================================
PROTOCOLS = {
    'red': {'dataset': 'red wine', 'set1': 23, 'set2': 42, 'set3': 69},
    'white': {'dataset': 'white wine', 'set1': 23, 'set2': 42, 'set3': 69},
    'houses': {'dataset': 'houses', 'set1': 23, 'set2': 42, 'set3': 69},
}

DATASETS = [
    'red wine',
    'white wine',
    'houses'
]

SUBSETS = [
    'set1',
    'set2',
    'set3',
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

    Parameters:
        dataset : string
            Name of the wanted dataset

    Returns:
        dataset : ndarray
            Contains the wanted dataset
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

    Parameters:
        reader : ReaderObject
            CSV reader object to read a file data
        skip_first_line : boolean
            Skip first line or not

    Returns:
        data : ndarray
            Array that contains the data
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

    Parameters:
        data : ndarray
            Contains the data
        subset : string
            The wanted subset from the list SUBSETS
        splits : dict
            The dict containing the list of subset and its random_state

    Returns:
        data dict : (ndarray, ndarray)
            Data split into (data_train, data_test)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a numpy array')
    if not isinstance(subset, str):
        raise TypeError('subset must be a str')
    if not isinstance(splits, dict):
        raise TypeError('splits must be a dict')
    if subset not in SUBSETS:
        raise ValueError(f'subset was not found in available subsets: {subset}')

    return train_test_split(data, train_size=0.5, test_size=0.5, random_state=splits[subset])


def get(protocol, subset, part):
    """Retrieve the wanted data subset as two numpy arrays X and Y

    Parameters:
        protocol : string
            Which protocol we want to use from PROTOCOLS (red, white, houses)
        subset : string
            The wanted subset from the list SUBSETS
        part : string
            The wanted part (either 'train' or 'test')

    Returns:
        data dict : (ndarray, ndarray)
            Two array X and Y
    """
    if not isinstance(protocol, str):
        raise TypeError('protocol must be a str')
    if not isinstance(subset, str):
        raise TypeError('subset must be a str')
    if not isinstance(part, str):
        raise TypeError('part must be a str')
    if protocol not in PROTOCOLS:
        raise ValueError(f'protocol was not found in available protocols: {protocol}')
    if subset not in SUBSETS:
        raise ValueError(f'subset was not found in available subsets: {subset}')
    if part not in ['train', 'test']:
        raise ValueError(f"part must either be 'train' or 'test': not {part}")

    fullData = split_data(load_dataset(
        PROTOCOLS[protocol]['dataset']), subset, PROTOCOLS[protocol])
    if part == 'train':
        wantedData = fullData[0]
    elif part == 'test':
        wantedData = fullData[1]
    else:
        raise ValueError(f"part must either be 'train' or 'test': not {part}")
    return wantedData.T[:wantedData.shape[1]-1].T, wantedData.T[-1].T.reshape(-1, 1)
