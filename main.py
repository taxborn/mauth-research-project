from matplotlib import pyplot as plt

import constants
import models
import preprocess
import multiprocessing
from validation import display_validations


def main():
    # Print out relevant parameters for the run
    print(f"{' mAuth Training Parameters ':-^{constants.MESSAGE_WIDTH}}")
    print("Current parameters:")
    print(f"Train/Test split: {100 * (1 - constants.TEST_SPLIT)}/{100 * constants.TEST_SPLIT}")
    print(f"Random state constant for reproducibility: {constants.RANDOM_STATE_CONSTANT}")
    print(f"Feature file used: {constants.FEATURE_FILE}")
    print(f"[Sequencing] length used: {constants.SEQUENCE_LENGTH}")
    print(f"[Anomaly detection] Percentage of negative (imposter) data: "
          f"{round(100 / constants.SPLIT, constants.NUM_ROUNDING)}%")
    if constants.FEATURES is not None:
        print(f"[Feature selection] Using the following features: {constants.FEATURES}")
    else:
        print(f"[Feature selection] Using all features generated in preprocessing.")
    print(f"[Models used]", end=" ")
    for model in constants.MODELS:
        print(f"{model}", end=" ")
    print(f"\n{' Starting Run ':-^{constants.MESSAGE_WIDTH}}\n")

    for model in constants.MODELS:
        model_roc_validations = []

        # SVC is single threaded, we can utilize the multiprocessing package to parallelize this. This is separate from
        # the other loop since we don't want to run this per-subject, we want this to be handled by the multiprocessing
        # package. If we are using GRIDSEARCH, we don't want to parallelize this since we want to use more resources for
        # GridSearch. There may be work here to where we can do 4 subjects at a time and dedicate 2 threads to GS per
        # subject, but that's too much work right now.
        if model == "svc" and not constants.USE_GRIDSEARCH:
            pool = multiprocessing.Pool(processes=10)
            subjects_to_process = range(constants.SUBJECTS)
            model_roc_validations = pool.map(models.parallel_svc, subjects_to_process)
            pool.close()
            pool.join()
            continue

        # Loop through the subjects
        for subject in range(constants.SUBJECTS):
            # Create the train/test split for the current subject. This creates a binary classifier.
            X_train, X_test, y_train, y_test = preprocess.binary_classify(constants.FEATURE_FILE, subject)
            predictions = None
            model_name = None
            clf = None

            if model == "knn":
                model_name = "k-Nearest Neighbors"
                predictions, clf = models.knn(X_train, X_test, y_train)
            elif model == "dt":
                model_name = "Decision Tree"
                predictions, clf = models.dt(X_train, X_test, y_train)
            elif model == "rf":
                model_name = "Random Forest"
                predictions, clf = models.rf(X_train, X_test, y_train)
            elif model == "svc":
                model_name = "Support Vector Classifier"
                predictions, clf = models.svc(X_train, X_test, y_train)
            else:
                print(f"Invalid model found")
                break

            # Check if both predictions and model name is set, otherwise break without validations since something
            # went wrong
            if predictions is None or not model_name or not clf:
                print("some sorta error happened buddy idk")
                break

            model_roc_validations.append(display_validations(X_test, y_test, predictions, model_name, subject, clf))

        for idx, value in enumerate(model_roc_validations):
            fpr, tpr, auc = value[0], value[1], value[2]
            plt.plot(fpr, tpr, label=f"AUC (#{idx}): {auc}")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"ROC Curve for all subjects on {model.upper()}")
        plt.xlabel("False Positive Rate (FPR)")
        plt.legend(loc=4)
        plt.show()


if __name__ == '__main__':
    main()
