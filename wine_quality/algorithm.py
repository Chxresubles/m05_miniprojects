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
from sklearn.tree import DecisionTreeClassifier
from utils import dataset, preprocessing, analysis


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================

def LR_train(train_X, train_Y):
    """Run training on the wine quality set using Linear Regression.

    Args:
        train_X (ndarray): Training set data of shape (n_samples, n_features)
        train_Y (ndarray): Training set targets of shape (n_samples, n_targets)

    Returns:
        model: Returns the trained LR model
    """
    lin_model = LinearRegression()
    lin_model.fit(train_X, train_Y)
    return lin_model


def LR_evaluate(model, test_X, test_Y):
    """Evaluate performance of the given Linear Regression model on the given set.

    Args:
        model (model): The trained model to run the prediction on.
        test_X (ndarray): Testing set data of shape (n_samples, n_features)
        test_Y (ndarray): Testing set targets of shape (n_samples, n_targets)
    """
    prediction = model.predict(test_X)
    MAE = analysis.MAE(prediction, test_Y)
    print(f"The mean absolute error of the model is: {MAE}\n")


def RT_train(train_X, train_Y):
    """Run training on the wine quality set using Regression Trees.

    Args:
        train_X (ndarray): Training set data of shape (n_samples, n_features)
        train_Y (ndarray): Training set targets of shape (n_samples, n_targets)

    Returns:
        model: Returns the trained RT model
    """
    reg_model = DecisionTreeClassifier(random_state=0)
    reg_model.fit(train_X, train_Y)
    return reg_model


def RT_evaluate(model, test_X, test_Y):
    """Evaluate performance of the given Regression Trees model on the given set.

    Args:
        model (model): The trained model to run the prediction on.
        test_X (ndarray): Testing set data of shape (n_samples, n_features)
        test_Y (ndarray): Testing set targets of shape (n_samples, n_targets)
    """
    prediction = model.predict(test_X).reshape(-1, 1)
    MAE = analysis.MAE(prediction, test_Y)
    print(f"The mean absolute error of the model is: {MAE}")


def trainAndTest(model_type, preprocess, subset, eval_set_str, color, poly):
    """Run training and testing on the chosen wine quality set using the wanted model type.

    Args:
        model_type (string): The model type to use (Can be either 'LR' or 'RT')
        preprocess (string): The preprocessing function to use (Can be either 'minmax' or 'znorm')
        subset (string): The set of train/test data to use
        eval_set_str (string): The set to use for evaluation (Can be either 'train' or 'test')
        color (string): The color of the wine (Can be either 'red' or 'white')
        poly (int): the maximal degree of the new polynomial features
                    (Call with 0 or 1 to not generate new polynomial features)
    """
    if color not in ['red', 'white']:
        raise ValueError(f'Color value was not recognized: {color}')
    # Load datasets
    print(f'Loading dataset {subset} ...')
    train_set = dataset.get(color, subset, 'train')
    test_set = dataset.get(color, subset, 'test')
    print('Dataset loaded')
    print('')

    train_set_X = train_set[0]
    train_set_Y = train_set[1]
    test_set_X = test_set[0]
    test_set_Y = test_set[1]

    # Data preprocessing
    # Generate polynomial features
    if poly > 1:
        print(f'Generating polynomial features with degree {poly}...')
        train_set_X = preprocessing.poly(train_set_X, poly)
        test_set_X = preprocessing.poly(test_set_X, poly)
        print('Done')
        print('')

    # Scale features
    print(f'Preprocessing data using {preprocess}...')
    if preprocess in 'minmax':
        train_set_X = preprocessing.min_max_scaling(train_set_X)
        test_set_X = preprocessing.min_max_scaling(test_set_X)
    elif preprocess == 'znorm':
        train_set_X = preprocessing.z_norm(train_set_X)
        test_set_X = preprocessing.z_norm(test_set_X)
    else:
        raise ValueError(f'Preprocessing value was not recognized: {preprocessing}')
    print('Preprocessing done')
    print('')

    # Prepare evaluation set
    if eval_set_str == 'test':
        eval_set_X = test_set_X
        eval_set_Y = test_set_Y
    elif eval_set_str == 'train':
        eval_set_X = train_set_X
        eval_set_Y = train_set_Y
    else:
        raise ValueError(f'Evaluation set name was not recognized: {eval_set_str}')

    # Model training and evaluation
    if model_type == 'LR':
        print(f'Training algorithm using Linear Regression over {train_set_X.shape[0]} lines of data...')
        model = LR_train(train_set_X, train_set_Y)
        print('Training done')
        print('')
        print(f'Evaluating algorithm using the {eval_set_str} set over {eval_set_X.shape[0]} lines of data...')
        LR_evaluate(model, eval_set_X, eval_set_Y)
        print('Evaluation done')
        print('')
    elif model_type == 'RT':
        print(f'Training algorithm using Regression Trees using {train_set_X.shape[0]} lines of data...')
        model = RT_train(train_set_X, train_set_Y)
        print('Training done')
        print('')
        print(f'Evaluating algorithm using the {eval_set_str} set over {eval_set_X.shape[0]} lines of data...')
        RT_evaluate(model, eval_set_X, eval_set_Y)
        print('Evaluation done')
        print('')
    else:
        raise ValueError(f'Model type value was not recognized: {model_type}')
