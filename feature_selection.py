import numpy as np
import scipy as sp
import pandas as pd
import feature_extraction
import common
import sklearn


def normalize_data(dataframe):
    """
    Removes bad data and
    applies a standardization to a DataFrame.

    :param dataframe: target
    :return: standardized DataFrame
    """
    dataframe_clean = dataframe.dropna()  # removing bad data first

    x = dataframe_clean.values
    feature_names = dataframe.columns
    N = x.shape[0]
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # excluding classes columns
    mean[len(feature_names) - len(common.classes):len(feature_names)] = 0
    std[len(feature_names) - len(common.classes):len(feature_names)] = 1

    # tile is used to keep the same dimensions
    mean = np.tile(mean, (N, 1))
    std = np.tile(std, (N, 1))

    # normalizing data with standardization
    # TODO: https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#standardizing-and-normalizing---how-it-can-be-done-using-scikit-learn
    x_norm = (x - mean) / std
    normalized_dataframe = pd.DataFrame(x_norm, columns=feature_names)

    # console output
    print('-- normalize_data completed --',
          'corrupted entries: ' + str(len(dataframe.index) - N),
          'updated entries: ' + str(N),
          sep="\n")

    return normalized_dataframe


def select_features(dataframe, n_feat):
    # TODO: fine tune parameters
    regularization_parameter = 1
    kernel = 'poly'
    poly_degree = 3

    # using smv as a classifier SVM
    classifier = sklearn.svm.SVC(
        C=regularization_parameter,
        kernel=kernel,
        degree=poly_degree
    )


    # selecting features
    selector = sklearn.feature_selection.RFE(classifier, n_feat)
    X, Y = common.dataframe_to_matrices(dataframe)
    selector.fit(X, Y)

    print('-- select_features completed --',
          'features selected: ' + str(n_feat),
          'rfe support: ',
          selector.support_,
          'rfe ranking: ',
          selector.ranking_,
          sep='\n')
