import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature(feature_name):
    """

    """
    # reading data
    dataframe_folder = os.path.join('resources', 'dataframes')
    dataframe_path = os.path.join(dataframe_folder, 'normalized_data.csv')
    dataframe = pd.read_csv(dataframe_path)
    x = dataframe.groupby('CLASS')[feature_name].plot(kind='density', grid=True)
    plt.show()
    print()


feature_list = ['zcr_mean', 'spectral_rolloff_mean']
for f in feature_list:
    plot_feature(f)
