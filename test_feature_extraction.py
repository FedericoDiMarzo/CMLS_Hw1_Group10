import os
import feature_extraction
from sklearn.utils import shuffle

dataframe_path = os.path.join('resources', 'dataframes')

dataframe = feature_extraction.extract_all()
dataframe = shuffle(dataframe)
dataframe.to_csv(os.path.join(dataframe_path, 'extracted_data.csv'), index=False)
