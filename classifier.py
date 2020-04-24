import common
import sklearn


def set_classifier(X_train, y_train, classifier_type='svm'):
    # console output
    print('-- set_classifier --',
          'classifier: ' + classifier_type,
          '',
          sep='\n')

    common.classifier = classifiers[classifier_type](X_train, y_train)
    common.classifier_type = classifier_type
    common.classifier.fit(X_train, y_train.ravel())


def test_classifier(X_test, y_test):
    # console output
    print('-- test_classifier --',
          'classifier: ' + common.classifier_type,
          'score: ' + str(common.classifier.score(X_test, y_test)),
          'confusion matrix: classical - country - disco - jazz ',
          str(sklearn.metrics.confusion_matrix(y_test,common.classifier.predict(X_test))),
          sep='\n')


def svm(X_train, Y_train):
    # console output
    print('C: ' + str(common.regularization_parameter),
          'kernel: ' + str(common.kernel),
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
