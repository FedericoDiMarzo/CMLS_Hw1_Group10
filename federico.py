import os
from feature_extraction import extract_all

csv_path = os.path.join('federico', 'test_data.csv')

data_frame = extract_all()
data_frame.to_csv(csv_path)
