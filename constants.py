# Constants used for data collection
DURATION = 60 * 30  # 60 seconds * 30 minutes
START_WAIT = 30
SUBJECT_ID = 0

# Constants used for data processing / training
# 0.15 = [96.288%], 0.20 = [96.281%], 0.33 = [96.223%], 0.5 = [96.2%]
TEST_SPLIT = 0.20
RANDOM_STATE = 0
# How many subjects of our 15 subject dataset to select from
SUBJECTS = 15
MESSAGE_WIDTH = 64
NUM_ROUNDING = 3
MODEL = "knn"  # DecisionTree, KNN, or SVC
