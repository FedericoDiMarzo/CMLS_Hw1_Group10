import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common import split_Xy

'''
This code provides : X_train_normalized, X_test_normalized, y_train, y_test

It also calculates X/y_train/test_i for each i class, for potential simulations
'''
def get_Xy_train_test():
    dataframe_mv = pd.read_csv('resources\\dataframes\\selected_features.csv')
    X, y = split_Xy(dataframe_mv.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_0 = X_train[y_train.ravel() == 0]
    X_train_1 = X_train[y_train.ravel() == 1]
    X_train_2 = X_train[y_train.ravel() == 2]
    X_train_3 = X_train[y_train.ravel() == 3]

    y_train_0 = np.zeros((X_train_0.shape[0],))
    y_train_1 = np.ones((X_train_1.shape[0],))
    y_train_2 = np.ones((X_train_2.shape[0],)) * 2
    y_train_3 = np.ones((X_train_3.shape[0],)) * 3

    X_test_0 = X_test[y_test.ravel() == 0]
    X_test_1 = X_test[y_test.ravel() == 1]
    X_test_2 = X_test[y_test.ravel() == 2]
    X_test_3 = X_test[y_test.ravel() == 3]

    y_test_0 = np.zeros((X_test_0.shape[0],))
    y_test_1 = np.ones((X_test_1.shape[0],))
    y_test_2 = np.ones((X_test_2.shape[0],)) * 2
    y_test_3 = np.ones((X_test_3.shape[0],)) * 3

    y_test = np.concatenate((y_test_0, y_test_1, y_test_2, y_test_3), axis=0)  #
    y_train = np.concatenate((y_train_0, y_train_1, y_train_2, y_train_3), axis=0)  #

    feat_max = np.max(np.concatenate((X_train_0, X_train_1, X_train_2, X_train_2, X_test_3), axis=0), axis=0)  #
    feat_min = np.min(np.concatenate((X_train_0, X_train_1, X_train_2, X_train_3), axis=0), axis=0)  #

    X_train_0_normalized = (X_train_0 - feat_min) / (feat_max - feat_min)
    X_train_1_normalized = (X_train_1 - feat_min) / (feat_max - feat_min)
    X_train_2_normalized = (X_train_2 - feat_min) / (feat_max - feat_min)
    X_train_3_normalized = (X_train_3 - feat_min) / (feat_max - feat_min)

    X_test_0_normalized = (X_test_0 - feat_min) / (feat_max - feat_min)
    X_test_1_normalized = (X_test_1 - feat_min) / (feat_max - feat_min)
    X_test_2_normalized = (X_test_2 - feat_min) / (feat_max - feat_min)
    X_test_3_normalized = (X_test_3 - feat_min) / (feat_max - feat_min)

    X_test_normalized = np.concatenate(
        (X_test_0_normalized, X_test_1_normalized, X_test_2_normalized, X_test_3_normalized), axis=0)  #
    X_train_normalized = np.concatenate(
        (X_train_0_normalized, X_train_1_normalized, X_train_2_normalized, X_train_3_normalized), axis=0)  #
    return(X_train_normalized, X_test_normalized, y_train, y_test)