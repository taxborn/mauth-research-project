import copy
import math

import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


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
    others = dataset['ID'] != subject
    return dataset[others].sample(samples, random_state=0)


def process(input: str):
    """
    Process a given CSV

    :return: X_train, X_val, y_train, y_val, where 50% is the target user and 50% is random data from other users.
    """
    dataset = pd.read_csv(input)
    print(f"Dataset loaded. Shape: {dataset.shape}")
    df = pd.DataFrame(dataset)
    num_features = int(dataset.shape[1])
    print(f"Number of features: {num_features}")
    X = dataset.values[:, 1:]
    y = dataset.values[:, 0]
    subject_data = glob.glob("data/*.csv")
    print(f"number of subjects: {len(subject_data)}")

    for subject in range(len(subject_data)):
        current_subject_data = df.loc[df.iloc[:, 0].isin([subject])]

        array_positive = copy.deepcopy(current_subject_data.values)
        array_positive[:, 0] = 1

        other_subject_data = get_other_data(dataset, subject, current_subject_data.shape[0])
        array_negative = copy.deepcopy(other_subject_data.values)
        array_negative[:, 0] = 0

        # concat the data
        mixed_set = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
        dump_dataset(pd.DataFrame(mixed_set), f"synth_data/user_{subject}_mixed_data.csv")
        X = mixed_set[:, 1:]
        y = mixed_set[:, 0]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=0)

        print(f"starting knn on {subject}")
        knn(X_train, X_val, y_train, y_val)


def knn(X_train, X_test, y_train, y_test):
    """
    K-Nearest Neighbor Model
    :return:
    """
    # Compute K. K = floor(sqrt(n)), and round K down to the nearest odd integer.
    n = len(y_train)
    k = int(math.sqrt(n))
    k += k % 2 - 1

    print(f"{' k-Nearest Neighbors ':-^50}")
    str = f" number of events: {n}, {k = } "
    print(f"{str:^50}")

    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    acc = round(100 * (tp + tn) / (tp + tn + fp + fn), 2)
    print(f"confusion matrix (accuracy: {acc}%):")
    print("[[ TP FN ]")
    print("[ FP TN ]]")
    print(f"{acc = } fnr = {fn / (fn + tp)} fpr = {fp / (fp + tn)}")
    print(cm)


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
    process("synth_data/user_all_data.csv", 1)


if __name__ == "__main__":
    main()
