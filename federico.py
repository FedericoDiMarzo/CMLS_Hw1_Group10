import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import extract_all

root_path = 'federico'

data_frame = extract_all()
data_frame.to_csv(os.path.join(root_path, 'test_data.csv'))

classical = data_frame[data_frame['classical'] == 1]
country = data_frame[data_frame['country'] == 1]
disco = data_frame[data_frame['disco'] == 1]
jazz = data_frame[data_frame['jazz'] == 1]

feature = 'spectral_decrease_mean'

sns.distplot(classical[feature])
sns.distplot(country[feature])
sns.distplot(disco[feature])
sns.distplot(jazz[feature])
plt.show()

