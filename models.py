import copy
import constants

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score


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

    # Get the current subjects' data, and update the 'ID' part to 1
    current_subject_data = df.loc[df.iloc[:, 0].isin([subject])]
    array_positive = copy.deepcopy(current_subject_data.values)
    array_positive[:, 0] = 1

    # Get the other subjects' data, and update the 'ID' part to 0
    other_subject_data = get_other_data(dataset, subject, current_subject_data.shape[0])
    array_negative = copy.deepcopy(other_subject_data.values)
    array_negative[:, 0] = 0

    # Concatenate the current subjects data and the other subjects data
    mixed_set = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
    dump_dataset(pd.DataFrame(mixed_set), f"synth_data/user_{subject}_mixed_data.csv")

    X = mixed_set[:, 1:]  # All the features
    y = mixed_set[:, 0]  # The subject ID is the first column

    # Returhn the split with constants defined at the top of the file
    return train_test_split(X, y, test_size=constants.TEST_SPLIT, random_state=constants.RANDOM_STATE)


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


def svm():
    """
     # select the columns to use as features
    features = ['mean_speed', 'std_speed', 'min_speed', 'max_speed', 'mean_acc', 'std_acc']

    split_start = time.time()

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['ID'], test_size=0.2, random_state=42)

    print(f"Split Complete")

    # create an SVM model
    model = SVC(kernel='linear')

    start_time = time.time()

    # train the model on the training data
    model.fit(X_train, y_train)

    # predict the labels of the test data
    y_pred = model.predict(X_test)

    # evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # print accuracy and time taken
    print(f"Accuracy: {accuracy}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    """


def dt(X_train, X_test, y_train):
    """
    Decision Tree Model
    :return:
    """
    model = DecisionTreeRegressor(random_state=constants.RANDOM_STATE)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def main():
    # Save timings for averages
    accuracies = []

    # Loop over the subjects
    for subject in range(constants.SUBJECTS):
        print(f"{f'k-Nearest Neighbors ({subject + 1} / {constants.SUBJECTS})':-^{constants.MESSAGE_WIDTH}}")
        # Split the dataset with the current subject
        X_train, X_test, y_train, y_test = process("synth_data/user_all_data.csv", subject)
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


if __name__ == "__main__":
    main()
