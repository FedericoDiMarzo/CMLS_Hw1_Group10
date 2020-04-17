import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import extract_all

root_path = 'federico'
class_list = ['classical', 'country', 'disco', 'jazz']

data_frame = extract_all()
data_frame.to_csv(os.path.join(root_path, 'test_data.csv'))

filtered_frame = [data_frame[data_frame[cls] == 1] for cls in class_list]

features = ['mfcc_3_mean', 'mfcc_3_std', 'mfcc_10_mean', 'mfcc_10_std']

for i, feat in enumerate(features):
    for frame in filtered_frame:
        plt.subplot(221+i)
        sns.distplot(frame[feat])

plt.show()
