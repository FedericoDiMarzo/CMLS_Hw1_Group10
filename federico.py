import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import extract_all
import feature_selection

root_path = 'federico'
class_list = ['classical', 'country', 'disco', 'jazz']

data_frame = extract_all()
data_frame = feature_selection.normalize_data(data_frame)
data_frame.to_csv(os.path.join(root_path, 'test_data.csv'))

feature_selection.select_features(data_frame, 2)

filtered_frame = [data_frame[data_frame[cls] == 1] for cls in class_list]
features = ['mfcc_2_mean', 'mfcc_3_mean', 'spectral_centroid_mean', 'spectral_decrease_mean']
for i, feat in enumerate(features):
    for frame in filtered_frame:
        plt.subplot(2, 2, 1 + i, label=features[i])
        sns.distplot(frame[feat])

plt.show()
