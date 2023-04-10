import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, \
    classification_report

import constants


def display_validations(X_test, y_test: np.ndarray, y_pred: np.ndarray, model: str, subject: int, classifier):
    """
    Generate validation for a given model output.

    :param y_test: The test split for model testing
    :param y_pred: The predicted output for X_test
    :param model: The name of the model, used for printing purposes
    :param subject: Which subject was used, also only used for printing
    :return: None
    """
    accuracy = round(accuracy_score(y_test, y_pred), constants.NUM_ROUNDING)
    precision = round(precision_score(y_test, y_pred), constants.NUM_ROUNDING)
    recall = round(recall_score(y_test, y_pred), constants.NUM_ROUNDING)
    f1 = round(f1_score(y_test, y_pred), constants.NUM_ROUNDING)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    far = round(fp / (tn + fn), constants.NUM_ROUNDING)
    frr = round(fn / (tp + fp), constants.NUM_ROUNDING)
    err = round((far + frr) / 2, constants.NUM_ROUNDING)

    print(f"\n{model} subject {subject} Validation details:")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    print(f"{cm}")
    print(f"FAR : {far} FRR: {frr} ERR: {err}")
    print(f"Report:\n{report}")
    print(f"{f' Finished {model} on subject {subject} ':-^{constants.MESSAGE_WIDTH}}")

    generate_roc(X_test, y_test, classifier)


def generate_roc(X_test, y_test, classifier):
    # y_pred_probability = classifier.predict_proba(X_test)[::, 1]
    # fpr, tpr, _ = roc_curve(y_test, y_pred_probability)
    # auc = round(roc_auc_score(y_test, y_pred_probability), constants.NUM_ROUNDING)
    # plt.plot(fpr, tpr, label=f"AUC: {auc}")
    # plt.ylabel("True Positive Rate")
    # plt.xlabel("False positive Rate")
    # plt.legend(loc=4)
    # plt.show()
    pass
