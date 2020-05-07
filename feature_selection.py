import numpy as np
import scipy as sp
import pandas as pd
import common
import classification
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.svm import SVC


def select_features(X, dataframe, verbose=False):
    """
    Variance selection of the features

    :param X: input data
    :param dataframe
    :param verbose: filters the console output
    :return: filter mask
    """
    original_features = len(dataframe.columns)

    # extracting y
    _, y = common.split_Xy(dataframe.values)

    # variance selection
    variance_selector = VarianceThreshold(common.min_var)
    variance_selector.fit(X, y)
    indices = variance_selector.get_support(indices=True)
    features_selected = [dataframe.columns[i] for i in np.nditer(indices)]
    X_new = variance_selector.transform(X)

    # console output
    if verbose:
        print('-- select_features completed--',
              'variance threshold: ' + str(common.min_var),
              'original features: ' + str(original_features),
              'filtered features: ' + str(len(features_selected)),
              '',
              'features selected:',
              sep='\n')

        for feat in features_selected:
            print(feat)
        print()  # just a newline for the console

    return variance_selector.get_support()


def print_feature_variance(X, dataframe):
    features = dataframe.columns[:-1]

    # calculating the variance for every feature
    variance = np.var(X, axis=0)

    # console output
    print('-- print_feature_variance completed--',
          'variance of every feature:',
          '',
          sep='\n')
    for index, f in enumerate(features):
        print(f + ': %0.4f' % variance[index])

    print()  # newline
