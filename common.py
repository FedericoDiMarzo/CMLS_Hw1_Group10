classes = ['classical', 'country', 'disco', 'jazz']

cep_start = 2
cep_end = 14
n_fft = 1024
win_length = 1024
hop_size = int(win_length / 2)
n_mels = 40
window = 'hann'
fmin = 133.33
fmax = 6853.8
n_feat = 10

classifier = None
regularization_parameter = 1
kernel = 'poly'
poly_degree = 3


def dataframe_to_matrices(dataframe):
    """
    Separates a dataframe into the matrix X, that contains
    the features, and Y, that contains the class value

    :param dataframe: target
    :return: (X, Y)
    """
    N = dataframe.shape[1]
    X = dataframe.values[:, :N - len(classes)]
    Y = dataframe.values[:, N - len(classes):]

    return X, Y
