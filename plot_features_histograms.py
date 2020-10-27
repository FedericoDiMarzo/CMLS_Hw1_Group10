'''
Here, you can plot features using plot_feature(list of 4 features)
examples :
plot_features(['onset_rate_low','onset_rate_high','chroma_max','chroma_min'])
plot_features(chroma_std(range(0,4)))
plot_features(chroma_std(range(0,4)))
'''

import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_normalization import clean_data
import pandas as pd

dataframe_folder = os.path.join('resources', 'dataframes')
dataframe_path = os.path.join(dataframe_folder, 'extracted_data.csv')
dataframe = pd.read_csv(dataframe_path)

# cleaning the whole dataset
dataframe = clean_data(dataframe)

colorlist = ["k", "r", "y", "b"]


def plot_features(features_list):
    # features = ['mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean']
    for f, feat in enumerate(features_list):
        for g in range(0, 4):
            frame = dataframe[dataframe.CLASS == g]
            plt.subplot(2, 2, 1 + f, label=features_list[f])
            sns.distplot(frame[feat], color=colorlist[g])
    plt.show()


def mfcc_mean(seq):
    feat = []
    for n in seq:
        feat.append("mfcc_" + str(n) + "_mean")
    return (feat)


def chroma_mean(seq):
    feat = []
    for n in seq:
        feat.append("chroma_" + str(n) + "_mean")
    return (feat)


def chroma_std(seq):
    feat = []
    for n in seq:
        feat.append("chroma_" + str(n) + "_std")
    return (feat)


def mfcc_std(seq):
    feat = []
    for n in seq:
        feat.append("mfcc_" + str(n) + "_std")
    return (feat)
