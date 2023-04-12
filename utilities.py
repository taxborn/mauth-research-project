import glob
import os
import shutil
import time
import constants
import numpy as np
from typing import Any
from sklearn.model_selection import GridSearchCV


def create_feature_file():
    """
    Creates the compound feature file, which is a file containing every single subjects' generated features compiled
    into one file

    :return: None
    """
    subject_data = glob.glob(f"synth_data/extracted_features_seq_{constants.SEQUENCE_LENGTH}/*.csv")
    subject_data.sort()  # glob lacks reliable ordering, so impose your own if output order matters

    # Check if the feature file exists already to avoid accidental overwrites. If the user doesn't answer
    # 'y' or 'yes', exit the script.
    # if os.path.exists(constants.FEATURE_FILE):
    #     answer = input(f"WARNING! {constants.FEATURE_FILE} exists. Do you want to overwrite? [Y/n]: ")
    #
    #     if answer.lower() is not "y" or answer.lower() is not "yes":
    #         print(f"k: {answer}")
    #         # We can just return early, and do nothing
    #         return

    with open(constants.FEATURE_FILE, 'wb') as outfile:
        for i, csv in enumerate(subject_data):
            with open(csv, 'rb') as infile:
                # If we are not in the first file, skip the header line
                if i != 0:
                    infile.readline()
                if i > 1:
                    outfile.write(b'\n')
                print(f"copying subject {csv}")
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)

    print(f"Finished. {constants.FEATURE_FILE} created.")


def find_best_classifier(model: str, classifier: Any, param_grid: dict[str, Any], X_train: np.ndarray,
                         y_train: np.ndarray) -> Any:
    """

    :param model: The name of the model, only for debug messages
    :param classifier: The classifier to base the GridSearch off of
    :param param_grid: A dictionary of hyperparameters and their possible values
    :param X_train: Feature training data
    :param y_train: Feature classifier data
    :return: The classifier with the best set of hyperparameters found. Those hyperparameters are also printed out.
    """
    print(f"> Starting GridSearch for {model}")
    gridsearch = GridSearchCV(classifier, param_grid, cv=constants.CROSS_VALIDATION_STEPS, n_jobs=constants.N_JOBS,
                              verbose=constants.VERBOSE)

    start_time = time.time()
    gridsearch.fit(X_train, y_train)
    print(f"> {model} GridSearch time: {round(time.time() - start_time, constants.NUM_ROUNDING)}s")

    best_params = gridsearch.best_params_
    print(f"> Best parameters for {model} were: {best_params}")
    return gridsearch.best_estimator_


if __name__ == "__main__":
    create_feature_file()
