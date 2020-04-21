"User can plot 4 features writting plot_features( ['mfcc_3_mean', 'an_other_feature_name'.. ] )"

import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import extract_all
import feature_selection

root_path = 'resources\dataframes'
class_list = ['classical', 'country', 'disco', 'jazz']

dataframe_path = os.path.join('resources', 'dataframes')

dataframe = extract_all()
dataframe.to_csv(os.path.join(dataframe_path, 'extracted_data.csv'), index=False)

colorlist=["k","r","y","b"]
def plot_features(features_list):
    #features = ['mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean']
    for f, feat in enumerate(features_list):
        for g in range(0,4):
            frame=dataframe[dataframe.CLASS == g]
            plt.subplot(2, 2, 1 + f, label=features_list[f])
            sns.distplot(frame[feat],color=colorlist[g])
    plt.show()

def mfcc_mean(seq):
    feat=[]
    for n in seq:
        feat.append("mfcc_" + str(n) + "_mean")
    return(feat)

def mfcc_std(seq):
    feat=[]
    for n in seq:
        feat.append("mfcc_" + str(n) + "_std")
    return(feat)