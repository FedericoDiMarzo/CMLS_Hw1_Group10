import os
from feature_normalization import clean_data, DataNormalizer
from sklearn.model_selection import train_test_split
from feature_selection import select_features, print_feature_variance
from classification import set_classifier, fit_classifier, test_classifier, cross_validation
import pandas as pd
import common
import time
import json

def set_configuration(configuration):
    common.classifier.kernel = configuration['kernel']
    common.classifier.C = configuration['regularization_parameter']
    common.classifier.poly_degree = configuration['poly_degree']
    common.min_var = configuration['min_var']

def get_configuration_score(configuration,dataframe, X, y, verbose=False):
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
    set_configuration(configuration)

    # testing the classifier with cross-validation :
    score = cross_validation(X, y, common.k_folds)
    #print(score)
    #print(common.classifier)

    #testing the classifier with standard validation :
    #X_train,X_val,y_train,y_val = train_test_split(X, y)
    #fit_classifier(X_train,y_train)
    #score = common.classifier.score(X_val, y_val)


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
    set_configuration(configuration)

    fit_classifier(X_train, y_train)

    # testing the classifier
    print('best parameters selected for the test : ',
          configuration,
          '',
          sep="\n")
    test_classifier(X_test, y_test)
