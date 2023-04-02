import copy
import math
import time

import constants

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from joblib import Parallel, delayed


def feature_set(df, feature_set_count=-1):
    """Used to loop through a fenerated feature list, otherwise will always dropna"""

    feature_sets = [
        ['ID', 'mean_x_speed', 'mean_y_speed', 'mean_speed', 'mean_x_acc', 'mean_y_acc'],
        ['ID', 'mean_x_speed', 'mean_y_speed', 'mean_speed', 'mean_x_acc', 'mean_y_acc', 'mean_acc', 'mean_jerk',
         'mean_ang',
         'mean_curve', 'mean_tan'],
        ['ID', 'std_x_speed', 'std_y_speed', 'std_speed', 'std_x_acc', 'std_y_acc', 'std_acc', 'std_ang', 'std_jerk',
         'std_curve', 'std_tan', 'min_tan']
    ]

    if feature_set_counter < 0:
        new_df = df[constants.FEATURE_LIST].copy()
        new_df.dropna(inplace=True)
        return new_df

    new_df = df[feature_set_count].copy()
    new_df.dropna(inplace=True)

    return new_df


def dump_dataset(dataset, outfile):
    dataset.to_csv(outfile)


def get_other_data(dataset, subject, samples):
    """
    Get a specified number of samples from other users. This will take a random sample from ALL the other
    subjects in the dataset. If you are targeting subject 7 who had 60907 events, it will take 60907
    events from other subjects, there might be some data from subject 1, 2, 8, etc..

    :param dataset: The dataset to take from
    :param subject: The current subject (to not take from)
    :param samples: The number of samples to take from the other subjects in the dataset
    :return:
    """
    other = dataset['ID'] != subject
    return dataset[other].sample(samples, random_state=constants.RANDOM_STATE)


def process(input: str, subject: int):
    """
    Process a given CSV and subject into a split where total data is from the 'genuine' or selected subject,
    and 50% of the data is a random sample from all the other subjects (excluding the genuine user).

    :return: X_train, X_val, y_train, y_val, for use in training models
    """
    print(f"> splitting data for subject {subject}")
    dataset = pd.read_csv(input)
    df = pd.DataFrame(dataset)
    df = feature_set(df)

    # Get the current subjects' data, and update the 'ID' part to 1
    current_subject_data = df.loc[df.iloc[:, 0].isin([subject])]
    array_positive = copy.deepcopy(current_subject_data.values)
    array_positive[:, 0] = 1

    # Get the other subjects' data, and update the 'ID' part to 0
    other_subject_data = get_other_data(dataset, subject, current_subject_data.shape[0])
    other_subject_data = feature_set(other_subject_data)
    array_negative = copy.deepcopy(other_subject_data.values)
    array_negative[:, 0] = 0

    # Concatenate the current subjects data and the other subjects data
    mixed_set = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
    # dump_dataset(pd.DataFrame(mixed_set), f"synth_data/user_{subject}_mixed_data.csv")

    X = mixed_set[:, 1:]  # All the features
    y = mixed_set[:, 0]  # The subject ID is the first column

    # Return the split with constants defined at the top of the file
    return train_test_split(X, y, test_size=constants.TEST_SPLIT, random_state=constants.RANDOM_STATE)


feature_set_counter = -1


def process_svc(input: str, subject: str):
    # Feature set counter, only convinient way to loop through list of feature lists.
    # global feature_set_counter
    # feature_set_counter += 1

    print(f"> splitting data for subjects: \n{subject}\n{input}")
    df = pd.read_csv(subject)
    df = feature_set(df)
    print(df.size)

    df2 = pd.read_csv(input)
    # df2 = df2.sample(n=(int(len(df)/10)))
    # df2 = feature_set(df2, feature_set_counter)
    print(df2.size)

    joined_sets = pd.concat([df, df2])

    joined_sets = feature_set(joined_sets)
    print(joined_sets.head())

    X = joined_sets.drop('ID', axis=1)
    y = joined_sets['ID']

    # Select the most seemingly relevant features using SelectKBes
    selector = SelectKBest(k=10)
    selector.fit(X, y)
    mask = selector.get_support()

    # Print the names of the selected features
    for col, selected in zip(X.columns, mask):
        if selected:

            print(col)

    X_selected = X.loc[:, mask]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)

    return train_test_split(X_scaled, y, test_size=.3,
                            random_state=constants.RANDOM_STATE)


feature_set_counter = -1


def knn(X_train, X_test, y_train) -> int:
    """
    :param X_train:
    :param X_test:
    :param y_train:
    :return: The accuracy of the classifier
    """
    # Compute K. K = floor(sqrt(n)), and round K down to the nearest odd integer.
    # This equation was taken from this YouTube video: https://www.youtube.com/watch?v=4HKqjENq9OU
    n = len(y_train)
    k = 3
    print(f"> number of events = {n}, {k = }")

    # Fit the classifier
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)


def svc_grid_search(X_train, X_test, y_train, y_test):
    # Reduce dimensionality with PCA
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Train an SVC classifier using parallel processing
    clf = SVC(C=1, kernel='rbf', random_state=constants.RANDOM_STATE, gamma='scale')
    # clf = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
    # clf.fit(X_train_pca, y_train)

    # Tune the hyperparameters
    # n_jobs = CORES PARALLELIZED
    param_grid = {'C': [0.1, 1], 'cache_size': [200, 400, 800]}
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=constants.N_JOBS)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Evaluate the model
    best_clf = grid_search.best_estimator_
    start_time = time.time()
    print(f"Prediction started at {start_time}")
    y_pred = best_clf.predict(X_test)
    print(f"Predict took took: {time.time() - start_time} minutes")
    return [classification_report(y_test, y_pred), best_clf, best_params]


def svc(X_train, X_test, y_train, y_test):
    start_time = time.time()
    clf = SVC(C=1, kernel='linear', random_state=constants.RANDOM_STATE)
    # clf = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
    clf.fit(X_train, y_train)
    print(f"Fit took: {(time.time() - start_time) / 60} minutes")

    start_time = time.time()
    y_pred = clf.predict(X_test)
    print(f"Predict took took: {(time.time() - start_time) / 60} minutes")


def dt(X_train, X_test, y_train):
    """
    Decision Tree Model
    :return:
    """
    start = time.time()
    # model = RandomForestClassifier(random_state=constants.RANDOM_STATE)
    model = DecisionTreeClassifier(random_state=constants.RANDOM_STATE)
    model.fit(X_train, y_train)
    print(f"Fit took: {(time.time() - start) / 60} minutes")
    start = time.time()
    prediction = model.predict(X_test)
    print(f"Fit took: {(time.time() - start) / 60} minutes")
    return prediction


def knn_run():
    # Save timings for averages
    accuracies = []

    # Loop over the subjects
    for subject in range(constants.SUBJECTS):
        print(f"{f'k-Nearest Neighbors ({subject + 1} / {constants.SUBJECTS})':-^{constants.MESSAGE_WIDTH}}")
        # Split the dataset with the current subject
        X_train, X_test, y_train, y_test = process("synth_data/extracted_features_data/user_all_extracted_data.csv",
                                                   subject)
        print(f"> starting knn...")

        # Run KNN which returns the accuracy of the model.
        y_pred = knn(X_train, X_test, y_train)

        ##############################################
        # Statistics about the model
        ##############################################
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (tn + fn)
        frr = fn / (tp + fp)
        err = (far + frr) / 2
        print(f"> {far = } {frr = } {err = }")
        # calculate accuracy
        acc = round(100 * (tp + tn) / (tp + tn + fp + fn), constants.NUM_ROUNDING)
        accuracies.append(acc)
        # calculate false positive rate, false negative rate
        fpr = round(100 * fn / (fn + tp), constants.NUM_ROUNDING)
        fnr = round(100 * fp / (fp + tn), constants.NUM_ROUNDING)
        print(f"> accuracy = {acc}% {fpr = }% {fnr = }%")

    acc_avg = round(sum(accuracies) / constants.SUBJECTS, constants.NUM_ROUNDING)

    # Print the statistics, this is the new f-string (or format string) syntax that was introduced in Python 3.6
    print(f"{' kNN Finished! ':-^{constants.MESSAGE_WIDTH}}")
    print(f"{f'average accuracy: {acc_avg}%':^{constants.MESSAGE_WIDTH}}")


def svc_run():
    df_13 = "synth_data/extracted_features_len_64_d2/user_13_extracted_64_d2.csv"
    # Get paths to loop over
    for i in range(10):
        file_name = f"user_{i}_extracted_64_d2.csv"
        file_path = os.path.join("synth_data/extracted_features_len_64_d2/", file_name)
        print(file_path)

        X_train, X_test, y_train, y_test = process_svc(file_path, df_13)
        print(f"> Starting grid search...")

        # Run SVC which returns the accuracy of the model.
        start_time = time.time()
        out = svc_grid_search(X_train, X_test, y_train, y_test)
        print(f"grid search took: {(time.time() - start_time)} minutes")

        ##############################################
        # Statistics about the model
        ##############################################
        if isinstance(out, list):
            print(f"Classification report: \n{out[0]}\nBest clf: {out[1]}\nBest params: {out[2]}")
        else:
            print(out)


def dt_run():
    accuracies = []
    for subject in range(constants.SUBJECTS):
        print(f"{f'Decision Tree ({subject + 1} / {constants.SUBJECTS})':-^{constants.MESSAGE_WIDTH}}")
        # Split the dataset with the current subject
        X_train, X_test, y_train, y_test = process("synth_data/extracted_features_data/user_all_extracted_data.csv",
                                                   subject)
        print(f"> Starting decision tree...")

        # Run KNN which returns the accuracy of the model.
        y_pred = dt(X_train, X_test, y_train)

        ##############################################
        # Statistics about the model
        ##############################################
        accuracies.append(accuracy_score(y_test, y_pred))
        print(accuracies)


if __name__ == "__main__":
    svc_run()
    # dt_run()
    # knn_run()
