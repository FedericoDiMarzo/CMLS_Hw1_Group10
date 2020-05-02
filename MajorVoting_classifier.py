import sklearn
from sklearn.model_selection import train_test_split
import sklearn.svm
import numpy as np
import pandas as pd
from common import split_Xy
import common
import os
from sklearn import multiclass

'BELOW : 2 EQUIVALENT METHODS   the n°1 builds manually 6 classifiers, the n°2 uses OneVsOne '

'CLASSIFIERS DEFINITION'
'1rst method : Manually designing 6 binary svm.vsc classifier : '
SVM_parameters={'C': common.regularization_parameter,'kernel' : 'rbf'}  #rbf gives better results than poly
clf_01 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_02 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_12 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_03 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_13 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_23 = sklearn.svm.SVC(**SVM_parameters, probability=True)

'2nd method : using one OneVsOne multiclass classifer'
OneVsOne = sklearn.multiclass.OneVsOneClassifier(sklearn.svm.SVC(kernel = 'rbf'))

'PROCESSING DATAS FOR THE 6 CLASSIFIERS'
dataframe_mv = pd.read_csv(os.path.join('resources', 'dataframes', 'selected_features.csv'))
X_mv,y_mv=split_Xy(dataframe_mv.values)
X_train, X_test, y_train, y_test = train_test_split(X_mv, y_mv)

X_train_0=X_train[y_train.ravel()==0]
X_train_1=X_train[y_train.ravel()==1]
X_train_2=X_train[y_train.ravel()==2]
X_train_3=X_train[y_train.ravel()==3]

y_train_0 = np.zeros((X_train_0.shape[0],))
y_train_1 = np.ones((X_train_1.shape[0],))
y_train_2 = np.ones((X_train_2.shape[0],))*2
y_train_3= np.ones((X_train_3.shape[0],))*3

X_test_0=X_test[y_test.ravel()==0]
X_test_1=X_test[y_test.ravel()==1]
X_test_2=X_test[y_test.ravel()==2]
X_test_3=X_test[y_test.ravel()==3]

y_test_0 = np.zeros((X_test_0.shape[0],))
y_test_1 = np.ones((X_test_1.shape[0],))
y_test_2 = np.ones((X_test_2.shape[0],))*2
y_test_3 = np.ones((X_test_3.shape[0],))*3

y_test_mc = np.concatenate((y_test_0, y_test_1, y_test_2, y_test_3), axis=0) #
y_train_mc = np.concatenate((y_train_0, y_train_1, y_train_2, y_train_3), axis=0) #

feat_max = np.max(np.concatenate((X_train_0, X_train_1, X_train_2, X_train_2, X_test_3), axis=0), axis=0) #
feat_min = np.min(np.concatenate((X_train_0, X_train_1, X_train_2, X_train_3), axis=0), axis=0) #

X_train_0_normalized = (X_train_0 - feat_min) / (feat_max - feat_min)
X_train_1_normalized = (X_train_1 - feat_min) / (feat_max - feat_min)
X_train_2_normalized = (X_train_2 - feat_min) / (feat_max - feat_min)
X_train_3_normalized = (X_train_3 - feat_min) / (feat_max - feat_min)

X_test_0_normalized = (X_test_0 - feat_min) / (feat_max - feat_min)
X_test_1_normalized = (X_test_1 - feat_min) / (feat_max - feat_min)
X_test_2_normalized = (X_test_2 - feat_min) / (feat_max - feat_min)
X_test_3_normalized = (X_test_3 - feat_min) / (feat_max - feat_min)

X_test_mc_normalized = np.concatenate((X_test_0_normalized, X_test_1_normalized, X_test_2_normalized ,X_test_3_normalized), axis=0) #
X_train_mc_normalized = np.concatenate((X_train_0_normalized, X_train_1_normalized, X_train_2_normalized, X_train_3_normalized), axis=0) #

'TRAINING'
'1rst method : '
clf_01.fit(np.concatenate((X_train_0_normalized, X_train_1_normalized), axis=0),np.concatenate((y_train_0, y_train_1), axis=0))
clf_02.fit(np.concatenate((X_train_0_normalized, X_train_2_normalized), axis=0),np.concatenate((y_train_0, y_train_2), axis=0))
clf_12.fit(np.concatenate((X_train_1_normalized, X_train_2_normalized), axis=0),np.concatenate((y_train_1, y_train_2), axis=0))
clf_03.fit(np.concatenate((X_train_0_normalized, X_train_3_normalized), axis=0),np.concatenate((y_train_0, y_train_3), axis=0))
clf_13.fit(np.concatenate((X_train_1_normalized, X_train_3_normalized), axis=0),np.concatenate((y_train_1, y_train_3), axis=0))
clf_23.fit(np.concatenate((X_train_2_normalized, X_train_3_normalized), axis=0),np.concatenate((y_train_2, y_train_3), axis=0))

'2nd method : '
OneVsOne.fit(X_train_mc_normalized,y_train_mc.ravel())


'PREDICTION'
y_test_predicted_01 = clf_01.predict(X_test_mc_normalized).reshape(-1, 1)
y_test_predicted_02 = clf_02.predict(X_test_mc_normalized).reshape(-1, 1)
y_test_predicted_12 = clf_12.predict(X_test_mc_normalized).reshape(-1, 1)
y_test_predicted_03 = clf_03.predict(X_test_mc_normalized).reshape(-1, 1)
y_test_predicted_13 = clf_13.predict(X_test_mc_normalized).reshape(-1, 1)
y_test_predicted_23  = clf_23.predict(X_test_mc_normalized).reshape(-1, 1)

y_test_predicted_mc = np.concatenate((y_test_predicted_01, y_test_predicted_02, y_test_predicted_12,y_test_predicted_03,y_test_predicted_13,y_test_predicted_23 ), axis=1)
y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)

y_test_predicted_mv = np.zeros((y_test_predicted_mc.shape[0],))
for i, e in enumerate(y_test_predicted_mc):
    y_test_predicted_mv[i] = np.bincount(e).argmax()

'2nd method :'
y_test_predicted_ovo=OneVsOne.predict(X_test_mc_normalized)


'METRICS'
def compute_cm_multiclass(gt, predicted):
    classes = np.unique(gt)

    CM = np.zeros((len(classes), len(classes)))

    for i in np.arange(len(classes)):
        pred_class = predicted[gt == i]

        for j in np.arange(len(pred_class)):
            CM[i, int(pred_class[j])] = CM[i, int(pred_class[j])] + 1
    print(CM)
def compute_metrics(gt_labels, predicted_labels):
    TP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 1))
    FP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 0))
    TN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 0))
    FN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 1))
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    print("Results : \n accuracy = {} \n precision = {} \n recall = {} \n F1 score = {}".format(
        accuracy, precision, recall, F1_score))
    return (F1_score)

print(' 1rst method: manually designing 6 classifiers')
compute_cm_multiclass(y_test_mc, y_test_predicted_mv)
compute_metrics(y_test_mc, y_test_predicted_mv)

print(' \n 2nd method : using OneVSOne')
compute_cm_multiclass(y_test_mc, y_test_predicted_ovo)
F1_score=compute_metrics(y_test_mc, y_test_predicted_ovo)
