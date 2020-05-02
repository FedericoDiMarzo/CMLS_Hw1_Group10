import os
from feature_normalization import clean_data, DataNormalizer
from sklearn.model_selection import train_test_split
from feature_selection import select_features
from classification import set_classifier, test_classifier, cross_validation
import pandas as pd
import numpy as np
import common
from Xy_train_test import get_Xy_train_test

dataframe_folder = os.path.join('resources', 'dataframes')

# reading data
dataframe_path = os.path.join(dataframe_folder, 'extracted_data.csv')
dataframe = pd.read_csv(dataframe_path)

# cleaning the whole dataset
dataframe = clean_data(dataframe)

# selecting the best features
# dataframe = select_features(dataframe, common.n_features)
# dataframe.to_csv(os.path.join(dataframe_folder, 'selected_features.csv'), index=False)

# building training and testing dataset
# X_train, X_test, y_train, y_test = get_Xy_train_test()

# separating data and normalization
X, y = common.split_Xy(dataframe.values)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X, _ = DataNormalizer(X).transform(X, X, common.normalization_type)
X_train, X_test = DataNormalizer(X_train).transform(X_train, X_test, common.normalization_type)

# defining the classifier
set_classifier(X_train, y_train)

# testing the classifier
test_classifier(X_test, y_test)
cross_validation(X, y, 10)
