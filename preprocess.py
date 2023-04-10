import copy
import numpy as np
import pandas as pd
import constants
import utilities
from sklearn.model_selection import train_test_split
from collections import deque
from scipy.signal import savgol_filter


def get_negative_data(dataset: pd.DataFrame, subject: int, num_of_samples: int) -> pd.DataFrame:
    """
    Get a specified number of samples from other users. This will take a random sample from ALL the other
    subjects in the dataset. If you are targeting subject 7 who had 60907 events, it will take 60907
    events from other subjects, there might be some data from subject 1, 2, 8, etc..

    :param dataset: The dataset to take from
    :param subject: The current subject (to not take from)
    :param num_of_samples: The number of samples to take from the other subjects in the dataset
    :return: A random sample of negative data, where negative data is any data that is not classified by the passed
    subject id
    """
    other = dataset['ID'] != subject

    return dataset[other].sample(num_of_samples, random_state=constants.RANDOM_STATE_CONSTANT)


def process(feature_file: str, subject: int):
    """
    Process a given CSV and subject into a split where total data is from the 'genuine' or selected subject,
    and 50% of the data is a random sample from all the other subjects (excluding the genuine user).

    :return: X_train, X_val, y_train, y_val, for use in training models
    """
    print(f"Starting processing for subject {subject}")
    dataset = pd.read_csv(feature_file)
    # Select only the relevant features we want from the constants file
    if constants.FEATURES is not None:
        dataset = dataset.loc[:, constants.FEATURES]

    # only print once
    if subject == 0:
        print(f"> features selected:\n{dataset.columns}")

    df = pd.DataFrame(dataset)
    pd.options.display.max_columns = None
    # Fill the NaNs with 0
    df.fillna(0, inplace=True)

    # Get the current subjects' data, and update the 'ID' part to 1
    current_subject_data = df.loc[df.iloc[:, 0].isin([subject])]
    array_positive = copy.deepcopy(current_subject_data.values)
    array_positive[:, 0] = 1

    # Get the other subjects' data, and update the 'ID' part to 0
    other_subject_data = get_negative_data(df, subject, int(current_subject_data.shape[0] / (constants.SPLIT - 1)))
    array_negative = copy.deepcopy(other_subject_data.values)
    array_negative[:, 0] = 0

    # Concatenate the current subjects data and the other subjects data
    mixed_set = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)])
    # If you want to output the binary classifiers, uncomment the following line
    # mixed_set.to_csv(f"synth_data/binary_classifiers/user_{subject}_mixed_data.csv")

    mixed_set = mixed_set.replace([np.inf, -np.inf], 0).to_numpy()

    X = mixed_set[:, 1:]  # All the features
    y = mixed_set[:, 0]  # The subject ID is the first column

    # Return the split with constants defined at the top of the file
    return train_test_split(X, y, test_size=constants.TEST_SPLIT, random_state=constants.RANDOM_STATE_CONSTANT)


def data_to_df(file_path):
    df = pd.read_csv(file_path)
    # Insert columns and run calculations
    df.insert(len(df.columns) - 1, "X_Speed", 0)
    df.insert(len(df.columns) - 1, "Y_Speed", 0)
    df.insert(len(df.columns) - 1, "Speed", 0)
    df.insert(len(df.columns) - 1, "X_Acceleration", 0)
    df.insert(len(df.columns) - 1, "Y_Acceleration", 0)
    df.insert(len(df.columns) - 1, "Acceleration", 0)
    df.insert(len(df.columns) - 1, "Jerk", 0)
    df.insert(len(df.columns) - 1, "Ang_V", 0)
    df.insert(len(df.columns) - 1, "Path_Tangent", 0)
    df.insert(len(df.columns) - 1, "Direction", 0)
    df = df.drop([0, 1]).reset_index(drop=True)

    df = df.loc[(df["X"].shift() != df["X"]) | (df["Y"].shift() != df["Y"])]  # Remove repeat data
    df['X_Speed'] = (df.X - df.X.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Y_Speed'] = (df.Y - df.Y.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Speed'] = np.sqrt((df.X_Speed ** 2) + (df.Y_Speed ** 2))
    df['X_Acceleration'] = (df.X_Speed - df.X_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Y_Acceleration'] = (df.Y_Speed - df.Y_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Acceleration'] = (df.Speed - df.Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Jerk'] = (df.Acceleration - df.Acceleration.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Path_Tangent'] = np.arctan2((df.Y - df.Y.shift(1)), (df.X - df.X.shift(1)))
    df['Ang_V'] = (df.Path_Tangent - df.Path_Tangent.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    # Fill empty data with 0
    df.fillna(0, inplace=True)
    # print(f"Size: {df.size} \nShape {df.shape} \nColumn Names: {df.columns}")
    df = sequence_maker(df)
    return df


def sequence_maker(df):
    sequential_data = []
    prev_data = deque(maxlen=constants.SEQUENCE_LENGTH)
    count = 0
    # Save ID
    ID = int(df.iloc[1]['ID'])
    # calculate the average of the 'values' column while omitting zeros
    button_non_zero_values = df.loc[df['Duration'] != 0, 'Duration']
    press_avg = button_non_zero_values.mean()

    for i in df.values:
        # Append each even row in df to prev_data without 'Subject ID' column, up to 60 rows
        prev_data.append([n for n in i[:-1]])
        if len(prev_data) == constants.SEQUENCE_LENGTH:
            temp = np.copy(prev_data)
            for j in range(7, 14):
                temp[0, j] = 0

            button_press_time = temp[1: 4].max()

            x_values = temp[1:, 2]
            y_values = temp[1:, 3]

            # Calculate the area under the curve using the trapezoidal rule
            area_under_curve = np.trapz(y_values, x_values)

            mean_x_speed = temp[1:, 5].mean()
            std_x_speed = temp[1:, 5].std()
            min_x_speed = temp[1:, 5].min()
            max_x_speed = temp[1:, 5].max()

            mean_y_speed = temp[1:, 6].mean()
            std_y_speed = temp[1:, 6].std()
            min_y_speed = temp[1:, 6].min()
            max_y_speed = temp[1:, 6].max()

            mean_speed = temp[1:, 7].mean()
            std_speed = temp[1:, 7].std()
            min_speed = temp[1:, 7].min()
            max_speed = temp[1:, 7].max()

            # Calculate mean speeds over distance
            dx = np.diff(temp[1:, 2])
            dy = np.diff(temp[1:, 3])
            dist = np.sqrt(dx ** 2 + dy ** 2)
            time = np.diff(temp[1:, 1])
            speed_over_dist = np.divide(dist ** 2, time)

            mean_speed_over_dist = np.mean(speed_over_dist)
            std_speed_over_dist = np.std(speed_over_dist)
            min_speed_over_dist = np.min(speed_over_dist)
            max_speed_over_dist = np.max(speed_over_dist)

            mean_x_acc = temp[1:, 8].mean()
            std_x_acc = temp[1:, 8].std()
            min_x_acc = temp[1:, 8].min()
            max_x_acc = temp[1:, 8].max()

            mean_y_acc = temp[1:, 9].mean()
            std_y_acc = temp[1:, 9].std()
            min_y_acc = temp[1:, 9].min()
            max_y_acc = temp[1:, 9].max()

            mean_acc = temp[1:, 10].mean()
            std_acc = temp[1:, 10].std()
            min_acc = temp[1:, 10].min()
            max_acc = temp[1:, 10].max()

            acceleration = np.divide(np.divide(dist, time), time)

            mean_acceleration_over_dist = np.mean(acceleration)
            std_acceleration_over_dist = np.std(acceleration)
            min_acceleration_over_dist = np.min(acceleration)
            max_acceleration_over_dist = np.max(acceleration)

            mean_jerk = temp[1:, 11].mean()
            std_jerk = temp[1:, 11].std()
            min_jerk = temp[1:, 11].min()
            max_jerk = temp[1:, 11].max()

            mean_ang = temp[1:, 12].mean()
            std_ang = temp[1:, 12].std()
            min_ang = temp[1:, 12].min()
            max_ang = temp[1:, 12].max()

            mean_tan = temp[1:, 13].mean()
            std_tan = temp[1:, 13].std()
            min_tan = temp[1:, 13].min()
            max_tan = temp[1:, 13].max()

            elapsed_time = temp[-1, 0] - temp[0, 0]
            # Initialize variables and data structures.
            curve_list = list()  # a list to store the curvature values for each segment of the trajectory
            traj_length = 0  # the total length of the trajectory
            accTimeatBeg = 0  # the accumulated time spent in acceleration at the beginning of the trajectory
            numCritPoints = 0  # the number of critical points (where the curvature is very low)
            path = list()  # a list to store the distances traveled for each segment of the trajectory
            flag = True  # a flag to indicate whether the trajectory is in the acceleration phase

            # Loop through each row in the input sequence.
            for k in range(1, constants.SEQUENCE_LENGTH):
                # Calculate the length of the trajectory segment between the current and previous rows and add it to
                # the list.
                traj_length += np.sqrt((temp[k, 1] - temp[k - 1, 1]) ** 2 + (temp[k, 2] - temp[k - 1, 2]) ** 2)
                path.append(traj_length)

                # Calculate the time and velocity differences between the current and previous rows.
                dt = temp[k, 0] - temp[k - 1, 0]
                dv = temp[k, 11] - temp[k - 1, 11]

                # If the velocity difference is positive and the trajectory is in the acceleration phase,
                # add the time difference to the accumulated time.
                if dv > 0 and flag:
                    accTimeatBeg += dt
                else:
                    flag = False

            # Loop through each segment of the trajectory.
            for ii in range(1, len(path)):
                # Calculate the distance and angle differences between the current and previous segments.
                dp = path[ii] - path[ii - 1]
                dangle = temp[ii, 12] - temp[ii - 1, 12]

                # Calculate the curvature of the segment and add it to the list.
                curv = dangle / dp
                curve_list.append(curv)

                # If the curvature is very low, increment the number of critical points.
                if abs(curv) < .0005:
                    numCritPoints += 1

            # Calculate the mean, standard deviation, minimum, and maximum curvatures from the list.
            mean_curve = np.mean(curve_list)
            std_curve = np.std(curve_list)
            min_curve = np.min(curve_list)
            max_curve = np.max(curve_list)

            sum_of_angles = np.sum(temp[1:, 12])
            # sharp_angles = np.sum(abs(temp[1:, 12]) < .0005)

            # Calculate smoothness
            t = temp[1:, 1]
            # calculate angle
            dt = np.diff(t)
            angle = temp[1:, 12]
            # smooth speed and angle
            smoothed_angle = savgol_filter(angle, window_length=5, polyorder=2, mode='mirror')
            # calculate derivative of smoothed angle
            d_smoothed_angle = np.diff(smoothed_angle) / dt
            # calculate smoothness
            mean_smoothness = np.abs(d_smoothed_angle).mean()
            std_smoothness = np.abs(d_smoothed_angle).std()
            min_smoothness = np.abs(d_smoothed_angle).min()
            max_smoothness = np.abs(d_smoothed_angle).max()

            elapsed_time = temp[-1, 0] - temp[0, 0]
            distance = np.sqrt((temp[-1, 2] - temp[0, 2]) ** 2 + (temp[-1, 3] - temp[0, 3]) ** 2)
            if distance != 0:
                straightness = traj_length / distance
            else:
                distance = np.sqrt((temp[-1, 2] - temp[0, 2]) ** 2 + (temp[-1, 3] - temp[0, 3]) ** 2)
                straightness = traj_length / distance

            for jj in [[mean_x_speed, mean_y_speed, mean_speed, mean_x_acc, mean_y_acc, mean_acc,
                        mean_jerk, mean_ang, mean_curve, mean_tan,
                        std_x_speed, std_y_speed, std_speed, std_x_acc, std_y_acc, std_acc,
                        std_ang, std_jerk, std_curve, std_tan, min_tan,
                        min_x_speed, min_y_speed, min_speed, min_x_acc, min_y_acc, min_acc,
                        min_ang, min_jerk, min_curve,
                        max_x_speed, max_y_speed, max_speed, max_x_acc, max_y_acc, max_acc,
                        max_ang, max_jerk, max_curve, max_tan, traj_length, numCritPoints,
                        mean_speed_over_dist, std_speed_over_dist,
                        min_speed_over_dist, max_speed_over_dist, mean_acceleration_over_dist,
                        std_acceleration_over_dist, max_acceleration_over_dist, min_acceleration_over_dist,
                        mean_smoothness, std_smoothness, min_smoothness, max_smoothness, area_under_curve]]:
                sequential_data.append(
                    jj)  # Prev_data now contains SEQ_LEN amount of samples and can be appended as one batch of 60 for RNN
        count += 1
        if count % 1000 == 0:
            print(count)
    df = pd.DataFrame(sequential_data,
                      columns=['mean_x_speed', 'mean_y_speed', 'mean_speed', 'mean_x_acc', 'mean_y_acc', 'mean_acc',
                               'mean_jerk', 'mean_ang', 'mean_curve', 'mean_tan',
                               'std_x_speed', 'std_y_speed', 'std_speed', 'std_x_acc', 'std_y_acc', 'std_acc',
                               'std_ang', 'std_jerk', 'std_curve', 'std_tan', 'min_tan',
                               'min_x_speed', 'min_y_speed', 'min_speed', 'min_x_acc', 'min_y_acc', 'min_acc',
                               'min_ang', 'min_jerk', 'min_curve',
                               'max_x_speed', 'max_y_speed', 'max_speed', 'max_x_acc', 'max_y_acc', 'max_acc',
                               'max_ang', 'max_jerk', 'max_curve', 'max_tan', 'traj_length', 'numCritPoints',
                               'mean_speed_over_dist', 'std_speed_over_dist',
                               'min_speed_over_dist', 'max_speed_over_dist', 'mean_acceleration_over_dist',
                               'std_acceleration_over_dist', 'max_acceleration_over_dist',
                               'min_acceleration_over_dist', 'mean_smoothness', 'std_smoothness', 'min_smoothness',
                               'max_smoothness', 'area_under_curve'])
    df.insert(0, 'ID', ID)
    df.fillna(0)
    # print(f"Head: {df.head()} \nSize: {df.size} \nShape {df.shape} \nColumn Names: {df.columns}")
    df.to_csv(f"synth_data/extracted_features_seq_{constants.SEQUENCE_LENGTH}/user_{ID}_extracted_{constants.SEQUENCE_LENGTH}.csv", index=False)
    return df


def process_subject(subject_num):
    _ = data_to_df(f"{constants.RAW_FOLDER_PATH}user_{subject_num}_data.csv")
    print(f"Finished processing subject {subject_num}")


if __name__ == "__main__":
    import multiprocessing

    num_processes = 16
    pool = multiprocessing.Pool(processes=num_processes)
    subjects_to_process = range(15)
    pool.map(process_subject, subjects_to_process)
    pool.close()
    pool.join()

    utilities.create_feature_file()
