import math
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import constants
import preprocess
from utilities import find_best_classifier
from validation import display_validations


def knn(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, find_best: bool = False) -> np.ndarray:
    """
    k-Nearest Neighbors Classifier.

    :param X_train: The training feature set.
    :param X_test: The testing feature set.
    :param y_train: The training subject IDs
    :param find_best: Option to use GridSearch to find the best classifier hyperparameters.
    :return: The predictions from the KNN Classifier and the classifier used
    """
    print(f"{' Starting KNN ':-^{constants.MESSAGE_WIDTH}}")
    # Compute K where K is the nearest odd integer from the square root of the length of the input data.
    n = len(y_train)
    k = int(math.sqrt(n))
    k += k % 2 - 1
    k = 3
    print(f"number of events: {n}, k = {k}")

    classifier = KNeighborsClassifier(n_neighbors=k, metric='cityblock', n_jobs=constants.N_JOBS)

    # KNN Grid search. Used for fine-tuning the hyperparameters.
    if find_best:
        param_grid = {'n_neighbors': [3, 5, 13, 27, k], 'metric': ['euclidean', 'cityblock']}
        classifier = find_best_classifier("KNN", classifier, param_grid, X_train, y_train)

    start_time = time.time()
    classifier.fit(X_train, y_train)
    print(f"> KNN Fit time: {round(time.time() - start_time, constants.NUM_ROUNDING)}s")

    start_time = time.time()
    y_pred = classifier.predict(X_test)
    took = time.time() - start_time
    eps = len(X_test) / took
    print(f"> KNN Predict time: {round(took, constants.NUM_ROUNDING)}s")
    print(f"> KNN Can process {round(eps, constants.NUM_ROUNDING)} events / second")

    return y_pred, classifier


def dt(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, find_best: bool = False) -> np.ndarray:
    """
    Decision Tree Classifier.

    :param X_train: The training feature set.
    :param X_test: The testing feature set.
    :param y_train: The training subject IDs
    :param find_best: Option to use GridSearch to find the best classifier hyperparameters.
    :return: The predictions from the Decision Tree Classifier and the classifier itself
    """
    print(f"{' Starting DT ':-^{constants.MESSAGE_WIDTH}}")
    classifier = DecisionTreeClassifier(random_state=constants.RANDOM_STATE_CONSTANT)

    # DT Grid search. Used for fine-tuning the hyperparameters.
    if find_best:
        param_grid = {'max_depth': [5, 7, 9, 10], 'min_samples_leaf': [2, 3, 5], 'min_samples_split': [5, 7, 9, 10],
                      'max_features': ["auto", "sqrt", "log2"]}
        classifier = find_best_classifier("Decision Tree", classifier, param_grid, X_train, y_train)

    start_time = time.time()
    classifier.fit(X_train, y_train)
    print(f"> Decision Tree Fit time: {round(time.time() - start_time, constants.NUM_ROUNDING)}s")

    start_time = time.time()
    y_pred = classifier.predict(X_test)
    took = time.time() - start_time
    eps = len(X_test) / took
    print(f"> Decision Tree Predict time: {round(took, constants.NUM_ROUNDING)}s")
    print(f"> Decision Tree Can process {round(eps, constants.NUM_ROUNDING)} events / second")
    return y_pred, classifier


def rf(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, find_best: bool = False) -> np.ndarray:
    """
    Random Forest Classifier.

    :param X_train: The training feature set.
    :param X_test: The testing feature set.
    :param y_train: The training subject IDs
    :param find_best: Option to use GridSearch to find the best classifier hyperparameters.
    :return: The predictions from the Random Forest Classifier and the classifier itself
    """
    print(f"{' Starting RF ':-^{constants.MESSAGE_WIDTH}}")
    classifier = RandomForestClassifier(random_state=constants.RANDOM_STATE_CONSTANT)

    # RF Grid search. Used for fine-tuning the hyperparameters.
    if find_best:
        param_grid = {}
        classifier = find_best_classifier("Random Forest", classifier, param_grid, X_train, y_train)

    start_time = time.time()
    classifier.fit(X_train, y_train)
    print(f"> Random Forest Fit time: {round(time.time() - start_time, constants.NUM_ROUNDING)}s")

    start_time = time.time()
    y_pred = classifier.predict(X_test)
    took = time.time() - start_time
    eps = len(X_test) / took
    print(f"> Random Forest Predict time: {round(took, constants.NUM_ROUNDING)}s")
    print(f"> Random Forest Can process {round(eps, constants.NUM_ROUNDING)} events / second")
    return y_pred, classifier


def svc(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, find_best: bool = False) -> np.ndarray:
    """
    Support Vector Classifier.


    :param X_train: The training feature set.
    :param X_test: The testing feature set.
    :param y_train: The training subject IDs
    :param find_best: Option to use GridSearch to find the best classifier hyperparameters.
    :return: The predictions from the Support Vector Classifier and the classifier itself
    """
    print(f"{' Starting SVC ':-^{constants.MESSAGE_WIDTH}}")
    classifier = SVC(C=100, gamma="auto", random_state=constants.RANDOM_STATE_CONSTANT)

    # SVC Grid search. Used for fine-tuning the hyperparameters.
    if find_best:
        param_grid = {'C': [1, 10, 100, 500], 'gamma': ["auto", "scale"]}
        classifier = find_best_classifier("SVC", classifier, param_grid, X_train, y_train)

    # Scale the data
    start_time = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"> Feature Scaling time: {round(time.time() - start_time, constants.NUM_ROUNDING)}s")

    start_time = time.time()
    classifier.fit(X_train_scaled, y_train)
    print(f"> SVC Fit time: {round(time.time() - start_time, constants.NUM_ROUNDING)}s")

    start_time = time.time()
    y_pred = classifier.predict(X_test_scaled)
    took = time.time() - start_time
    eps = len(X_test_scaled) / took
    print(f"> SVC Predict time: {round(took, constants.NUM_ROUNDING)}s")
    print(f"> SVC Can process {round(eps, constants.NUM_ROUNDING)} events / second")
    return y_pred, classifier


def parallel_svc(subject: int):
    """
    A parallelized wrapper around SVC for multiprocessing.

    :param subject: The subject ID
    :return: None
    """
    X_train, X_test, y_train, y_test = preprocess.process(constants.FEATURE_FILE, subject)
    y_pred, classifier = svc(X_train, X_test, y_train)
    display_validations(X_test, y_test, y_pred, "SVC", subject, classifier)
