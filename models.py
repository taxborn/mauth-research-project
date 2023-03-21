import time
import copy
import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

TEST_SPLIT = 0.20  # 0.15 = [96.288%], 0.20 = [96.281%], 0.33 = [96.223%], 0.5 = [96.2%]
RANDOM_STATE = 0
SUBJECTS = 15  # How many subjects of our 15 subject dataset to select from


def dump_dataset(dataset, outfile):
    dataset.to_csv(outfile)


def get_other_data(dataset, subject, samples):
    """
    Get a specified number of samples from other users

    :param dataset: The dataset to take from
    :param subject: The current subject (to not take from)
    :param samples: The number of samples to take from the negatives
    :return:
    """
    return dataset[dataset['ID'] != subject].sample(samples, random_state=RANDOM_STATE)


def process(input: str, subject: int):
    """
    Process a given CSV

    :return: X_train, X_val, y_train, y_val, where 50% is the target user and 50% is random data from other users.
    """
    print(f"splitting data for subject {subject}")
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

    # Concatonate the current subjects data and the other subjects data
    # TODO: If you dump this, gives weird headers.
    mixed_set = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
    # dump_dataset(pd.DataFrame(mixed_set), f"synth_data/user_{subject}_mixed_data.csv")

    X = mixed_set[:, 1:]
    y = mixed_set[:, 0]

    return train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE)


def knn(X_train, X_test, y_train, y_test):
    """
    K-Nearest Neighbor Model
    :return:
    """
    # Compute K. K = floor(sqrt(n)), and round K down to the nearest odd integer.
    n = len(y_train)
    k = int(math.sqrt(n))
    k += k % 2 - 1

    # Print debug information
    str = f" number of events: {n}, {k = } "
    print(f"{str:^50}")

    # Fit the classifier
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Statistics about the model
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    tp = cm[0][0]  # true positive
    fn = cm[0][1]  # false negative
    fp = cm[1][0]  # false positive
    tn = cm[1][1]  # true negative
    acc = round(100 * (tp + tn) / (tp + tn + fp + fn), 3)
    # print(f"confusion matrix (accuracy: {acc}%):")
    # print("[[ TP FN ]")
    # print("[ FP TN ]]")
    print(f"{acc = }% fnr = {fn / (fn + tp)} fpr = {fp / (fp + tn)}")

    return acc


def svm():
    """
    Support Vector Machine / C-Support Vector Classification Model
    :return:
    """
    pass


def dt():
    """
    Decision Tree Model
    :return:
    """
    pass


def main():
    split_times = []
    model_times = []
    accuracies = []

    tot_start = time.time()
    for subject in range(SUBJECTS):
        print(f"{' k-Nearest Neighbors ':-^50}")
        str = f"round {subject + 1} / {SUBJECTS}"
        print(f"{str:^50}")
        start = time.time()
        X_train, X_test, y_train, y_test = process("synth_data/user_all_data.csv", subject)
        end = time.time()
        took = end - start
        split_times.append(took)

        print(f"split finished (took {round(took, 3)}s). starting knn...")
        start = time.time()
        acc = knn(X_train, X_test, y_train, y_test)
        end = time.time()
        accuracies.append(acc)
        took = end - start
        model_times.append(took)
        print(f"knn took {round(took, 3)}s.")

    tot_end = time.time()
    tot = round(tot_end - tot_start, 3)
    sp_avg = round(sum(split_times) / SUBJECTS, 3)
    md_avg = round(sum(model_times) / SUBJECTS, 3)
    acc_avg = round(sum(accuracies) / SUBJECTS, 3)
    print(f"{' kNN Finished! ':-^64}")
    str = f"average split time: {sp_avg}s, average model time: {md_avg}s"
    print(f"{str:^64}")
    str = f"average accuracy: {acc_avg}%"
    print(f"{str:^64}")
    str = f"total train and validation time: {tot}s"
    print(f"{str:^64}")


if __name__ == "__main__":
    main()
