classes = ['classical', 'country', 'disco', 'jazz']


def dataframe_to_matrices(dataframe):
    """
    Separates a dataframe into the matrix X, that contains
    the features, and Y, that contains the class value

    :param dataframe: target
    :return: (X, Y)
    """
    N = dataframe.shape[0]
    X = dataframe.values[:, :N - len(classes)]
    Y = dataframe.values[N - len(classes), N]

    return X, Y
