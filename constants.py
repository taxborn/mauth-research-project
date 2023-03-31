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

N_JOBS = -1  # -1 for all cores, otherwise specify number of cores

FEATURE_LIST = ['ID', 'mean_x_speed', 'mean_y_speed', 'mean_speed', 'mean_x_acc', 'mean_y_acc', 'mean_acc',
                               'mean_jerk', 'mean_ang',
                               'mean_curve', 'mean_tan',
                               'std_x_speed', 'std_y_speed', 'std_speed', 'std_x_acc', 'std_y_acc', 'std_acc',
                               'std_ang', 'std_jerk',
                               'std_curve', 'std_tan', 'min_tan',
                               'min_x_speed', 'min_y_speed', 'min_speed', 'min_x_acc', 'min_y_acc', 'min_acc',
                               'min_ang', 'min_jerk',
                               'min_curve',
                               'max_x_speed', 'max_y_speed', 'max_speed', 'max_x_acc', 'max_y_acc', 'max_acc',
                               'max_ang', 'max_jerk',
                               'max_curve', 'max_tan',
                               'elapsed_time', 'sum_of_angles', 'accTimeatBeg', 'traj_length', 'numCritPoints',
                               'button_press_time']