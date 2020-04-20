import numpy as np
import scipy as sp
import pandas as pd
import common
import sklearn.feature_selection


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
        Extract mean and std from train_data, in order to
        prepare for normalization

        :param train_data: target
        """
        x = train_data.values
        feature_names = train_data.columns
        N = x.shape[0]
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        # excluding classes columns
        self.mean[len(feature_names) - len(common.classes):len(feature_names)] = 0
        self.std[len(feature_names) - len(common.classes):len(feature_names)] = 1

        # tile is used to keep the same dimensions
        self.mean = np.tile(self.mean, (N, 1))
        self.std = np.tile(self.std, (N, 1))

    def transform(self, dataframe):
        """
        Applying standardization to a DataFrame
        # TODO: https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#standardizing-and-normalizing---how-it-can-be-done-using-scikit-learn

        :param dataframe: target
        :return: normalized DataFrame
        """
        X = dataframe.values
        X_trasformed = (X - self.mean) / self.std
        return pd.DataFrame(X_trasformed, columns=dataframe.columns)
