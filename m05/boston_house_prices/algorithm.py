

"""
ML algorithms for Boston House Price dataset.

This script contains two ML algorithms for predicting Boston House Price
depending of the first 13 variables rows. For the prediciton we will use

- Linear regression
- Regression trees
 """


# ============================================================================================================
# Imports
# ============================================================================================================
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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
    reg_model = DecisionTreeRegressor(random_state=0)
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
    print(f"The mean absolute error of the model is: {MAE}\n")


def trainAndTest(model_type, preprocess, subset, eval_set_str, poly):
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
    # Load datasets
    train_set = dataset.get('houses', subset, 'train')
    test_set = dataset.get('houses', subset, 'test')

    train_prices = train_set[1]
    train_features = train_set[0]
    test_prices = test_set[1]
    test_features = test_set[0]

    print(numpy.shape(train_prices))
    print(numpy.shape(train_features))
    print(numpy.shape(test_prices))
    print(numpy.shape(test_features))

    if poly > 1:
        print(f'Generating polynomial features with degree {poly}...')
        train_features = preprocessing.poly(train_features, poly)
        test_features = preprocessing.poly(test_features, poly)
        print('Done')
        print('')

    # Data preprocessing, choose the method between minmax and znorm
    if preprocess in 'minmax':
        train_features = preprocessing.min_max_scaling(train_features)
        test_features = preprocessing.min_max_scaling(test_features)
    elif preprocess == 'znorm':
        train_features = preprocessing.z_norm(train_features)
        test_features = preprocessing.z_norm(test_features)
    else:
        raise ValueError(
            f'Preprocessing value was not recognized: {preprocessing}')

    # Prepare evaluation set
    if eval_set_str == 'test':
        eval_set_features = test_features
        eval_set_prices = test_prices
    elif eval_set_str == 'train':
        eval_set_features = train_features
        eval_set_prices = train_prices
    else:
        raise ValueError(
            f'Evaluation set name was not recognized: {eval_set_str}')

    # Model training and evaluation
    if model_type == 'LR':
        model = LR_train(train_features, train_prices)
        LR_evaluate(model, eval_set_features, eval_set_prices)
    elif model_type == 'RT':
        model = RT_train(train_features, train_prices)
        RT_evaluate(model, eval_set_features, eval_set_prices)
    else:
        raise ValueError(f'Model type value was not recognized: {model_type}')


def main():
    """Main function to be called from the command-line"""

    import argparse

    example_doc = """\
Examples:
    1. Train a Linear Regression algorithm
       and test it on the test set:
       $ boston_house_prices --model LR --evalset test
    2. Train a Regression Tree algorithm
       and test it on the train set:
       $ boston_house_prices --model RT --evalset train
    3. Train a Linear Regression algorithm using z-normalisation
       and by creating 2-degree polynomial features
       and test it on the test set:
       $ boston_house_prices --model LR --preprocess znorm --subset set1 --evalset test --poly 2
    """

    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Train and test the wanted ML algorithm on the boston house prices dataset",
        epilog=example_doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=["LR", "RT"],
        default="LR",
        help="Choose the model type to train and test."
             " Either 'LR' for Linear Regression"
             " or 'RT' for Regression Trees",
    )

    parser.add_argument(
        "-p",
        "--preprocess",
        choices=["minmax", "znorm"],
        default="minmax",
        help="Choose the preprocessing to be applied on the data."
             " Either 'minmax' or 'znorm'",
             )

    parser.add_argument(
        "-s",
        "--subset",
        choices=["set1", "set2", "set3"],
        default="set1",
        help="Choose the data separation (train/test)"
             " to be applied on the dataset."
             " Either 'set1', 'set2' or 'set3'",
             )

    parser.add_argument(
        "-e",
        "--evalset",
        choices=["train", "test"],
        default="test",
        help="Choose the data set to be used"
             " for evaluation."
             " Either 'train' or 'test'",
    )

    parser.add_argument(
        "-n",
        "--poly",
        type=int,
        default=0,
        help="Set the number of polynomial features"
             " to be created with the input data.",
        )

    args = parser.parse_args()

    # Run the script with the input arguments
    trainAndTest(args.model, args.preprocess, args.subset, args.evalset, args.poly)
