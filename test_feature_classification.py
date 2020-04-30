import os
from feature_normalization import clean_data, DataNormalizer
from sklearn.model_selection import train_test_split
from feature_selection import select_features
from classifier import set_classifier, test_classifier
import pandas as pd
import common
from Xy_train_test import get_Xy_train_test

dataframe_folder = os.path.join('resources', 'dataframes')

# reading data
dataframe_path = os.path.join(dataframe_folder, 'extracted_data.csv')
dataframe = pd.read_csv(dataframe_path)

# cleaning and normalizing the whole dataset
dataframe = clean_data(dataframe)
dataframe = DataNormalizer(dataframe).transform(dataframe)



# selecting the best features
dataframe = select_features(dataframe)
dataframe.to_csv(os.path.join(dataframe_folder, 'selected_features.csv'))

'1 - building training and testing set with preprocessing method'
from sklearn import preprocessing
X, y = common.split_Xy(dataframe.values)
X_train, X_test, y_train, y_test = train_test_split(X, y)
std_scale = preprocessing.MinMaxScaler().fit(X_train) #or replace MinMax by Standard
X_train_norm = std_scale.transform(X_train)
X_test_norm = std_scale.transform(X_test)

'''
2- building training and testing dataset with X-_train_test
X_train_norm,X_test_norm,y_train,y_test=get_Xy_train_test()
'''

# defining the classifier
set_classifier(X_train_norm, y_train, classifier_type='svm')

# testing the classifier
test_classifier(X_test_norm, y_test)
