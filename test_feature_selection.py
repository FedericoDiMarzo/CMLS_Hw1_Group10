import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import extract_all
import sklearn
import pandas as pd
import feature_selection
import feature_extraction
import common

dataframe_path = os.path.join('resources', 'dataframes', 'normalized_data.csv')
dataframe = pd.read_csv(dataframe_path)

common.classifier = sklearn.svm.SVC(
        C=common.regularization_parameter,
        kernel=common.kernel,
        degree=common.poly_degree
    )
feature_selection.select_features(dataframe, common.n_feat)

