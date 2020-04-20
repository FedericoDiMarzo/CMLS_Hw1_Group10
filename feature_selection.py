import numpy as np
import scipy as sp
import pandas as pd
import common
from sklearn.feature_selection import VarianceThreshold


def select_features(dataframe):
    """
    Variance selection of the features

    :param dataframe: input dataframe
    :return: filtered dataframe
    """
    original_features = len(dataframe.columns)

    # extracting X and y
    X, y = common.split_Xy(dataframe.values)

    # variance selection
    variance_selector = VarianceThreshold(common.min_var)
    variance_selector.fit(X, y)
    indices = variance_selector.get_support(indices=True)
    features_selected = [dataframe.columns[i] for i in np.nditer(indices)]
    X_new = variance_selector.transform(X)

    # console output
    print('-- select_features completed--',
          'variance threshold: ' + str(common.min_var),
          'original features: ' + str(original_features),
          'filtered features: ' + str(len(features_selected)),
          '',
          'features selected:',
          sep='\n')

    for feat in features_selected:
        print(feat)

    print()  # just a newrow for the console
    new_columns = features_selected + ['CLASS']
    new_dataframe = pd.DataFrame(common.merge_Xy(X_new, y),
                                 columns=new_columns)
    return new_dataframe
