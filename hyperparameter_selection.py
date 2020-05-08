import os
from feature_normalization import clean_data, DataNormalizer
from sklearn.model_selection import train_test_split
from feature_selection import select_features, print_feature_variance
from classification import set_classifier, fit_classifier, test_classifier, cross_validation
import pandas as pd
import common
import time
import json


def get_configuration_score(dataframe, X, y, verbose=False):
    timer_start = time.time()

    # normalization
    X, _ = DataNormalizer(X).transform(X, X, common.normalization_type)

    # printing features variance
    if verbose:
        print_feature_variance(X, dataframe)

    # selecting the best features
    feature_mask = select_features(X, dataframe, verbose=verbose)
    X = X[:, feature_mask]

    # defining the classifier
    set_classifier(common.classifier_type)

    # testing the classifier on the training set
    score = cross_validation(X, y, common.k_folds)



    # console output
    if verbose:
        execution_time = time.time() - timer_start
        print('',
              '-- test_feature_classification completed --',
              'execution time: ' + '{:.2f}'.format(execution_time) + 's',
              '',
              sep="\n")

    return score


def final_test(dataframe, X_train, y_train, X_test, y_test):
    best_parameters_path = os.path.join('resources', 'parameters', 'best_parameters.json')
    with open(best_parameters_path, 'r') as file:
        configuration = json.load(file)

    common.set_configuration(configuration)

    # normalization
    X_train, X_test = DataNormalizer(X_train).transform(X_train, X_test, common.normalization_type)

    # printing features variance
    print_feature_variance(X_train, dataframe)

    # selecting the best features
    feature_mask = select_features(X_train, dataframe, verbose=True)
    X_train = X_train[:, feature_mask]
    X_test = X_test[:, feature_mask]

    # defining the classifier
    set_classifier(common.classifier_type, verbose=True)
    fit_classifier(X_train, y_train)

    # testing the classifier
    test_classifier(X_test, y_test)
