# Constants for data collection
DURATION = 60 * 30  # 60 seconds * 30 minutes
START_WAIT = 30
SUBJECT_ID = 0

# Constants for data preprocessing and training
TEST_SPLIT = 0.30
RANDOM_STATE_CONSTANT = 0

SUBJECTS = 15
MESSAGE_WIDTH = 53
NUM_ROUNDING = 3
# Which model(s) to run, either "dt", "rf", "knn", "svc"
MODELS = ["knn"]

# -1 for all cores, otherwise specify number of cores. If you're unsure, either 1 or 4 should do well here.
N_JOBS = -1

SEQUENCE_LENGTH = 128
CROSS_VALIDATION_STEPS = 5
USE_GRIDSEARCH = False

# What split we should use for anomaly detection, where the number here
# represents 1/n of the data should be 'imposter' data. For example, 2
# here means 1/2 of the dataset would be imposter data. 3 would mean 1/3
# is imposter data, and so on...
SPLIT = 2
VERBOSE = True

"""
Select the features used in the models. The current total set is:

mean_x_speed, mean_y_speed, mean_speed, mean_x_acc, mean_y_acc, 
mean_acc, mean_jerk, mean_ang, mean_curve, mean_tan, std_x_speed, 
std_y_speed, std_speed, std_x_acc, std_y_acc, std_acc, std_ang, 
std_jerk, std_curve, std_tan, min_tan, min_x_speed, min_y_speed, 
min_speed, min_x_acc, min_y_acc, min_acc, min_ang, min_jerk, 
min_curve, max_x_speed, max_y_speed, max_speed, max_x_acc, max_y_acc,
max_acc, max_ang, max_jerk, max_curve, max_tan, traj_length, 
numCritPoints, mean_speed_over_dist, std_speed_over_dist, 
min_speed_over_dist, max_speed_over_dist, mean_acceleration_over_dist,
std_acceleration_over_dist, max_acceleration_over_dist, 
min_acceleration_over_dist, mean_smoothness, std_smoothness, 
min_smoothness, max_smoothness, area_under_curve

The only required field is 'ID' and at least one other feature.
If this is set to None, the models will use all of the features
"""
FEATURES = None
# FEATURES = ["ID", "area_under_curve", "std_y_speed", "mean_y_speed", "std_smoothness", "mean_smoothness",
#             "std_acceleration_over_dist", "std_curve", "mean_curve", "std_acc", "mean_acc"]
FEATURE_FILE = f"./synth_data/user_all_features_SQ{SEQUENCE_LENGTH}.csv"
RAW_FOLDER_PATH = f"./raw_data/"
