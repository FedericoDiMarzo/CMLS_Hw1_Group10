import os
from feature_normalization import clean_data, DataNormalizer
from sklearn.model_selection import train_test_split
from feature_selection import select_features, print_feature_variance
from classification import set_classifier, test_classifier, cross_validation
import hyperparameter_selection as hs
import itertools
import pandas as pd
import common
import json
import time


# contains all the parameter combinations that we want to try
hyperparameters = {
    'min_var': [0.005, 0.01, 0.015, 0.2],
    'classifier_type': ['svm'],
    'regularization_parameter': [-2, -1, 0, 1, 2],
    'kernel': ['linear', 'poly'],
    'poly_degree': [2, 3, 4, 5]
}

optimize_hyperparameters = True

# reading data
dataframe_folder = os.path.join('resources', 'dataframes')
dataframe_path = os.path.join(dataframe_folder, 'extracted_data.csv')
dataframe = pd.read_csv(dataframe_path)

# cleaning the whole dataset
dataframe = clean_data(dataframe)

# separating data
X, y = common.split_Xy(dataframe.values)
X, X_validation, y, y_validation = train_test_split(X, y)

# this part of the script can be really slow, thus it can
# be turned off with the flag
if optimize_hyperparameters:
    parameters_timer_start = time.time()
    best_parameters = {
        'score': 0
    }

    best_parameters_path = os.path.join('resources', 'parameters', 'best_parameters.json')

    # finds the best parameters
    iteration_counter = 0
    hyperparameters_list = [dict(zip(hyperparameters.keys(), v))
                            for v in itertools.product(*hyperparameters.values())]
    for configuration in hyperparameters_list:
        # console output
        iteration_counter += 1
        print('[[iteration n.' + str(iteration_counter) + ']]')

        # sets the configuration
        common.set_configuration(configuration)

        # calculate the score of the configuration
        score = hs.get_configuration_score(dataframe, X, y)

        if score > best_parameters['score']:
            # updates the best parameters
            best_parameters['score'] = score
            for parameter, value in configuration.items():
                best_parameters[parameter] = value
    print()  # newline

    # stores the best parameters
    with open(best_parameters_path, mode='w') as file:
        json.dump(best_parameters, file)

    # console output
    parameters_execution_time = time.time() - parameters_timer_start
    print('-- parameters optimization completed --',
          'execution time: ' + '{:.2f}'.format(parameters_execution_time) + 's',
          'parameters selected: ',
          best_parameters,
          sep="\n")

# evaluating on validation set
hs.validate_configuration(dataframe, X, y, X_validation, y_validation)