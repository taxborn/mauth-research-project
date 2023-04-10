import constants
import models
import preprocess
import multiprocessing
from validation import display_validations


def main():
    # Print out relevant parameters for the run
    print("mAuth Models Run.")
    print("Current parameters:")
    print(f"Train/Test split: {100 * (1 - constants.TEST_SPLIT)}/{100 * constants.TEST_SPLIT}")
    print(f"Random state constant for reproducibility: {constants.RANDOM_STATE_CONSTANT}")
    print(f"Feature file used: {constants.FEATURE_FILE}")
    print(f"[Sequencing] length used: {constants.SEQUENCE_LENGTH}")
    print(f"[Anomaly detection] Percentage of negative (imposter) data: "
          f"{round(100 / constants.SPLIT, constants.NUM_ROUNDING)}%")

    for model in constants.MODELS:
        # SVC is single threaded, we can utilize the multiprocessing package to parallelize this. This is separate from
        # the other loop since we don't want to run this per-subject, we want this to be handled by the multiprocessing
        # package.
        if model == "svc":
            pool = multiprocessing.Pool(processes=10)
            subjects_to_process = range(constants.SUBJECTS)
            pool.map(models.parallel_svc, subjects_to_process)
            pool.close()
            pool.join()
            continue

        # Loop through the subjects
        for subject in range(constants.SUBJECTS):
            # Create the train/test split for the current subject. This creates a binary classifier.
            X_train, X_test, y_train, y_test = preprocess.process(constants.FEATURE_FILE, subject)
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
            else:
                print(f"Invalid model found")
                break

            # Check if both predictions and model name is set, otherwise break without validations since something
            # went wrong
            if predictions is None or not model_name or not clf:
                print("some sorta error happened buddy idk")
                break

            display_validations(X_test, y_test, predictions, model_name, subject, clf)


if __name__ == '__main__':
    main()
