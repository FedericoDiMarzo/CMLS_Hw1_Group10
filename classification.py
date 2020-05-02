import common
import sklearn
import numpy as np
from sklearn.model_selection import StratifiedKFold


def set_classifier(X_train, y_train, classifier_type='svm'):
    """
    Sets the classifier

    :param X_train: train data
    :param y_train: train lables
    :param classifier_type: choosen classifier
    """
    # console output
    print('-- set_classifier --',
          'classifier: ' + classifier_type,
          sep='\n')

    common.classifier = classifiers[classifier_type]()
    common.classifier.fit(X_train, y_train.ravel())


def test_classifier(X_test, y_test):
    """
    Tries the classifier on test data

    :param X_test: test data
    :param y_test: test labels
    """
    # console output
    print('-- test_classifier --',
          'score: ' + str(common.classifier.score(X_test, y_test)),
          'confusion matrix: classical - country - disco - jazz ',
          str(sklearn.metrics.confusion_matrix(y_test, common.classifier.predict(X_test))),
          '',
          sep='\n')


def cross_validation(X, y, k=10):
    """
    K fold cross validation

    :param X: whole data
    :param y: whole lables
    :param k: number of subdivisions
    """

    scores = sklearn.model_selection.cross_val_score(
        estimator=common.classifier,
        X=X,
        y=y.flatten(),
        cv=k,
    )
    # console output
    print('-- cross_validation --',
          "accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2),
          '',
          sep='\n')


def svm():
    # console output
    print('C: ' + str(common.regularization_parameter),
          'kernel: ' + str(common.kernel) +
          str(common.poly_degree) if common.kernel == 'poly' else '',
          '',
          sep='\n')

    return sklearn.svm.SVC(
        C=common.regularization_parameter,
        kernel=common.kernel,
        degree=common.poly_degree
    )


classifiers = {
    'svm': svm
}
