import os
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import extract_all
import feature_selection

root_path = 'resources\dataframes'
class_list = ['classical', 'country', 'disco', 'jazz']

data_frame = extract_all()
data_frame = feature_selection.normalize_data(data_frame)
data_frame.to_csv(os.path.join(root_path, 'normalized_data.csv'))

filtered_frame = [data_frame[data_frame[cls] == 1] for cls in class_list]
colorlist=["k","r","y","b"]
def plot_features(features_list):
    #features = ['mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean']
    for f, feat in enumerate(features_list):
        for g,frame in enumerate(filtered_frame):
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