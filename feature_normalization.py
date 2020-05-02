import numpy as np
from sklearn import preprocessing

import Xy_train_test


def clean_data(dataframe):
    """
    Drops corrupted entries from the DataFrame

    :param dataframe: target
    :return: clean DataFrame
    """
    dataframe_clean = dataframe.dropna()
    # console output
    print('-- clean_data completed --',
          'removed entries: ' + str(len(dataframe.index) - len(dataframe_clean.index)),
          'total entries: ' + str(len(dataframe_clean.index)),
          '',
          sep="\n")
    return dataframe_clean


class DataNormalizer:

    def __init__(self, train_data):
        """
        Extracts mean, std, min_feat, max_feat from train_data, in order to
        prepare for normalization

        :param train_data: target
        """

        # min_feat and max_feat are used for minmax2
        self.min_feat = np.min(train_data, axis=0)
        self.max_feat = np.max(train_data, axis=0)

        # defining all the possible Scalers
        self.minmax_scaler = preprocessing.MinMaxScaler().fit(train_data)
        self.standardization_scaler = preprocessing.StandardScaler().fit(train_data)

    def transform(self, X_train, X_test, type='minmax'):
        """
        Applying normalization to a DataFrame
        # TODO: https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#standardizing-and-normalizing---how-it-can-be-done-using-scikit-learn

        :param X_train
        :param X_test
        :param type: type of normalization
        :return: (X_train_transformed, X_test_transformed, y_train, y_test)
        """
        X_train_transformed = None
        X_test_transformed = None

        if type == 'minmax':
            X_train_transformed = self.minmax_scaler.transform(X_train)
            X_test_transformed = self.minmax_scaler.transform(X_test)
        elif type == 'minmax2':
            X_train_transformed = (X_train - self.min_feat) / (self.max_feat - self.min_feat)
            X_test_transformed = (X_test - self.min_feat) / (self.max_feat - self.min_feat)
        elif type == 'standardization':
            X_train_transformed = self.standardization_scaler.transform(X_train)
            X_test_transformed = self.standardization_scaler.transform(X_test)
        else:
            # console output
            print('-- DataNormalizer.transform error --',
                  'type not recognized',
                  '',
                  sep="\n")

        return X_train_transformed, X_test_transformed
