import numpy as np
import sys

current_module = sys.modules[__name__]

classes = ['classical', 'country', 'disco', 'jazz']

cep_start = 2  # mel coefficient start
cep_end = 14  # mel coefficient end
n_fft = 2 ** 10  # lenght fft
win_length = n_fft  # window lenght
hop_size = int(win_length / 2)  # hop size
window = 'hann'  # type of window

n_mels = 40  # order of mel filter
fmin = 133.33  # mel filter freq min
fmax = 6853.8  # mel filter freq max

min_var = 0.015  # variance threshold for feats

normalization_type = 'minmax'  # type of normalization

k_folds = 10  # number of keyfolds

classifier = None  # classifier used
classifier_type = 'svm'  # string for the classifier used
regularization_parameter = 1  # regularization parameter for the classifier
kernel = 'linear'  # type of kernel for SVM
poly_degree = 3  # order of the polynomial for poly kernel


def split_Xy(Z):
    """
    Splits feature matrix and class vector
    """
    X = Z[:, :-1]
    y = Z[:, -1:]
    return X, y


def merge_Xy(X, y):
    """
    Merges X and Y together

    :param X: feature matrix
    :param y: output matrix
    :return: merged matrix
    """
    return np.concatenate((X, y), axis=1)
