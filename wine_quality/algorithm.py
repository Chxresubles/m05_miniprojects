#!/usr/bin/env python

"""
ML algorithms for the wine quality dataset.

This script contains two ML algorithms for
predicting wine quality in functions of
the wine intrinsic properties:
- Linear regression
- Regression trees
"""


# ============================================================================================================
# Imports
# ============================================================================================================

from sklearn.linear_model import LinearRegression
from utils import dataset, preprocessing, analysis


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================

def LR_train(train_X, train_Y):
    """Run training on the wine quality set using Linear Regression."""
    lin_model = LinearRegression()
    lin_model.fit(train_X, train_Y)
    return lin_model


def LR_evaluate(model, test_X, test_Y):
    """Evaluate performance of the given Linear Regression model on the given set."""
    prediction = model.predict(test_X)
    MAE = analysis.MAE(prediction, test_Y)
    print(f"The mean absolute error of the model is: {MAE}\n")


def RT_train(train_X, train_Y):
    """Run training on the wine quality set using Regression Trees."""
    # RT training
    pass


def RT_evaluate(model, test_X, test_Y):
    """Evaluate performance of the given Regression Trees model on the given set."""
    # RT evaluation on the set
    pass


def trainAndTest(model_type, preproc, eval_set_str, color):
    """Run training and testing on the chosen wine quality set using the wanted model type."""
    # Load datasets
    train_set = dataset.get(color, 'train')
    test_set = dataset.get(color, 'test')

    train_set_X = train_set[0]
    train_set_Y = train_set[1]
    test_set_X = test_set[0]
    test_set_Y = test_set[1]

    # Data preprocessing
    if preproc in ('minmax', None):
        train_set_X = preprocessing.min_max_scaling(train_set_X)
        train_set_Y = preprocessing.min_max_scaling(train_set_Y)
        test_set_X = preprocessing.min_max_scaling(test_set_X)
        test_set_Y = preprocessing.min_max_scaling(test_set_Y)
    elif preproc == 'znorm':
        train_set_X = preprocessing.z_norm(train_set_X)
        train_set_Y = preprocessing.z_norm(train_set_Y)
        test_set_X = preprocessing.z_norm(test_set_X)
        test_set_Y = preprocessing.z_norm(test_set_Y)
    else:
        raise ValueError(f'Preprocessing value was not recognized: {preprocessing}')

    # Prepare evaluation set
    if eval_set_str in ('test', None):
        eval_set_X = test_set_X
        eval_set_Y = test_set_Y
    elif eval_set_str == 'train':
        eval_set_X = train_set_X
        eval_set_Y = train_set_Y
    else:
        raise ValueError(f'Evaluation set name was not recognized: {eval_set_str}')

    # Model training and evaluation
    if model_type in ('LR', None):
        model = LR_train(train_set_X, train_set_Y)
        LR_evaluate(model, eval_set_X, eval_set_Y)
    elif model_type == 'RT':
        model = RT_train(train_set_X, train_set_Y)
        RT_evaluate(model, eval_set_X, eval_set_Y)
    else:
        raise ValueError(f'Model type value was not recognized: {model_type}')

