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
import numpy
import csv
import re
from sklearn.model_selection import train_test_split


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

WINE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wine_quality', 'dataset')

HOUSE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'boston_house_prices', 'dataset')


# ============================================================================================================
# Useful functions
# ============================================================================================================

def load_dataset(dataset):
    """Load the wanted dataset in a numpy array."""
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
        with open(os.path.join(HOUSE_FILE_PATH, 'housing.data'), 'rt') as f, open(os.path.join(HOUSE_FILE_PATH, 'housing.csv'), 'wt') as fcsv:
            for line in f:
                str = re.sub(' +', ';', line)
                str = re.sub(' \n', '\n', str)
                str.lstrip()
                fcsv.write(str[1:])
        with open(os.path.join(HOUSE_FILE_PATH, 'housing.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=';')
            data = load(reader, False)
    return data

def load(reader, skip_first_line):
    """Load content of the reader in a numpy array."""
    data = []
    for k, row in enumerate(reader):
        if not k and skip_first_line: continue
        data.append(numpy.array([float(z) for z in row]))
    data = numpy.vstack(data)
    return data


def split_data(data, subset, splits):
    """Get the wanted subset from the data as a numpy array."""
    return data[splits[subset]]


def get(protocol, subset):
    """Retrieve the wanted data subset as a numpy array."""
    retval = split_data(load_dataset(PROTOCOLS[protocol]['dataset']), subset, PROTOCOLS[protocol])
    return retval
