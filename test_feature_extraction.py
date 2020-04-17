import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import extract_all
import pandas as pd
import feature_selection
import feature_extraction

dataframe_path = os.path.join('resources', 'dataframes')

dataframe = feature_extraction.extract_all()
dataframe = feature_selection.normalize_data(dataframe)
dataframe.to_csv(os.path.join(dataframe_path, 'normalized_data.csv'))
