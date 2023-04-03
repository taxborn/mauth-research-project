import time
import numpy as np
import pandas as pd
from collections import deque
import os.path

import constants
import utils


def sequence_maker(df, sequence_length=64):
    sequential_data = []
    prev_data = deque(maxlen=sequence_length)
    count = 0
    # Save ID
    ID = int(df.iloc[1]['ID'])
    # calculate the average of the 'values' column while omitting zeros
    button_non_zero_values = df.loc[df['Duration'] != 0, 'Duration']
    press_avg = button_non_zero_values.mean()

    for i in df.values:
        prev_data.append(
            [n for n in i[:-1]])  # Append each row in df to prev_data without 'Subject ID' column, up to 60 rows
        if len(prev_data) == sequence_length:
            temp = np.copy(prev_data)
            for j in range(7, 14):
                temp[0, j] = 0

            button_press_time = temp[1: 4].max()
            if button_press_time == 0:
                button_press_time = press_avg

            # Calculate mean speeds over distance
            dx = np.diff(temp[1:, 2])
            dy = np.diff(temp[1:, 3])
            dist = np.sqrt(dx ** 2 + dy ** 2)
            time = np.diff(temp[1:, 1])
            speed_over_dist = np.divide(dist, time)

            mean_speed_over_dist = np.mean(speed_over_dist)
            std_speed_over_dist = np.std(speed_over_dist)
            min_speed_over_dist = np.min(speed_over_dist)
            max_speed_over_dist = np.max(speed_over_dist)

            acceleration = np.divide(np.diff(speed_over_dist), time[:-1])

            mean_acceleration_over_dist = np.mean(acceleration)
            std_acceleration_over_dist = np.std(acceleration)
            min_acceleration_over_dist = np.min(acceleration)
            max_acceleration_over_dist = np.max(acceleration)



            elapsed_time = temp[-1, 0] - temp[0, 0]
            # Initialize variables and data structures.
            curve_list = list()  # a list to store the curvature values for each segment of the trajectory
            traj_length = 0  # the total length of the trajectory
            accTimeatBeg = 0  # the accumulated time spent in acceleration at the beginning of the trajectory
            numCritPoints = 0  # the number of critical points (where the curvature is very low)
            path = list()  # a list to store the distances traveled for each segment of the trajectory
            flag = True  # a flag to indicate whether the trajectory is in the acceleration phase

            # Loop through each row in the input sequence.
            for k in range(1, sequence_length):
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

            for jj in [[mean_x_speed, mean_y_speed, mean_speed, mean_x_acc, mean_y_acc, mean_acc, mean_jerk, mean_ang,
                        mean_curve, mean_tan,
                        std_x_speed, std_y_speed, std_speed, std_x_acc, std_y_acc, std_acc, std_ang, std_jerk,
                        std_curve, std_tan, min_tan,
                        min_x_speed, min_y_speed, min_speed, min_x_acc, min_y_acc, min_acc, min_ang, min_jerk,
                        min_curve,
                        max_x_speed, max_y_speed, max_speed, max_x_acc, max_y_acc, max_acc, max_ang, max_jerk,
                        max_curve, max_tan,
                        elapsed_time, sum_of_angles, accTimeatBeg, traj_length, numCritPoints, button_press_time,
                        mean_speed_over_dist, std_speed_over_dist, min_speed_over_dist, max_speed_over_dist,
                        mean_acceleration_over_dist, std_acceleration_over_dist, max_acceleration_over_dist,
                        min_acceleration_over_dist]]:
                sequential_data.append(
                    jj)  # Prev_data now contains SEQ_LEN amount of samples and can be appended as one batch of 60 for RNN
        count += 1
        if count % 1000 == 0:
            print(count)
    df = pd.DataFrame(sequential_data,
                      columns=['mean_x_speed', 'mean_y_speed', 'mean_speed', 'mean_x_acc', 'mean_y_acc', 'mean_acc',
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
                               'button_press_time', 'mean_speed_over_dist', 'std_speed_over_dist',
                               'min_speed_over_dist', 'max_speed_over_dist', 'mean_acceleration_over_dist',
                               'std_acceleration_over_dist', 'max_acceleration_over_dist',
                               'min_acceleration_over_dist'])
    df.insert(0, 'ID', ID)
    df.fillna(0)
    print(f"Head: {df.head()} \nSize: {df.size} \nShape {df.shape} \nColumn Names: {df.columns}")
    df.to_csv(f"synth_data/extracted_features_len_64_d2/user_{ID}_extracted_{sequence_length}_d2.csv", index=False)
    return df


def subject_to_dataframe(path: str, remove_duplicates: bool = True) -> pd.DataFrame:
    print(f"Converting {path} into a mAuth subject dataframe...")
    # Check if it is a valid path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The subject path {path} was not found.")

    # Create dataframe
    df = pd.read_csv(path)

    # Insert columns needed for feature calculations
    df.insert(len(df.columns) - 1, "X_Speed", 0)
    df.insert(len(df.columns) - 1, "Y_Speed", 0)
    df.insert(len(df.columns) - 1, "Speed", 0)
    df.insert(len(df.columns) - 1, "X_Acceleration", 0)
    df.insert(len(df.columns) - 1, "Y_Acceleration", 0)
    df.insert(len(df.columns) - 1, "Acceleration", 0)
    df.insert(len(df.columns) - 1, "Jerk", 0)
    df.insert(len(df.columns) - 1, "Path_Tangent", 0)
    df.insert(len(df.columns) - 1, "Ang_V", 0)
    # df.insert(len(df.columns) - 1, "Direction", 0)

    # Remove rows where the next row has the exact same X and Y value, for example, if we had:
    # BEFORE:
    #   .. | X | Y | ...
    #   .. | 2 | 5 | ...
    #   .. | 3 | 6 | ...
    #   .. | 3 | 6 | ...
    #   .. | 3 | 5 | ...
    # Then after this line the dataframe would become:
    # AFTER:
    #   .. | X | Y | ...
    #   .. | 2 | 5 | ...
    #   .. | 3 | 6 | ...
    #   .. | 3 | 5 | ...
    # Removing the duplicate (3, 6).
    if remove_duplicates:
        df = df.loc[(df["X"].shift() != df["X"]) | (df["Y"].shift() != df["Y"])]
        print(" > removed duplicate rows")

    # Now populate the inserted columns
    print(" > Populating new columns used for features")
    start = time.time()
    df['X_Speed'] = (df.X - df.X.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Y_Speed'] = (df.Y - df.Y.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Speed'] = np.sqrt((df.X_Speed ** 2) + (df.Y_Speed ** 2))
    df['X_Acceleration'] = (df.X_Speed - df.X_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Y_Acceleration'] = (df.Y_Speed - df.Y_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Acceleration'] = (df.Speed - df.Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Jerk'] = (df.Acceleration - df.Acceleration.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Path_Tangent'] = np.arctan2((df.Y - df.Y.shift(1)), (df.X - df.X.shift(1)))
    df['Ang_V'] = (df.Path_Tangent - df.Path_Tangent.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    took = round(time.time() - start, constants.NUM_ROUNDING)
    print(f" > Done! ({took =}s)")

    return df


def split(dataframe: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    sequences = []
    ID = int(dataframe.iloc[1]['ID'])

    print(f"Splitting subject {ID} into sequences ({n = }).")

    start = time.time()
    for idx in range(0, dataframe.shape[0], n):
        sequence = dataframe.iloc[idx:idx + n]
        sequences.append(sequence)

    took = round(time.time() - start, constants.NUM_ROUNDING)
    print(f" > {took = }s")

    return sequences

def process(sequences: list[pd.DataFrame]) -> pd.DataFrame:
    dfs = []
    start_time = time.time()
    for count, sequence in enumerate(sequences):
        sql_data = []
        """ Calculate start/end time and how long the sequence lasts """
        sequence_start, sequence_end = sequence['Timestamp'][~sequence['Timestamp'].isna()].values[[0, -1]]
        elapsed = sequence_end - sequence_end

        """ Calculate speeds """
        mean_x_speed = sequence['X_Speed'].mean()
        std_x_speed = sequence['X_Speed'].std()
        min_x_speed = sequence['X_Speed'].min()
        max_x_speed = sequence['X_Speed'].max()

        mean_y_speed = sequence['Y_Speed'].mean()
        std_y_speed = sequence['Y_Speed'].std()
        min_y_speed = sequence['Y_Speed'].min()
        max_y_speed = sequence['Y_Speed'].max()

        mean_speed = sequence['Speed'].mean()
        std_speed = sequence['Speed'].std()
        min_speed = sequence['Speed'].min()
        max_speed = sequence['Speed'].max()

        """ Calculate accelerations """
        mean_x_acc = sequence['X_Acceleration'].mean()
        std_x_acc = sequence['X_Acceleration'].std()
        min_x_acc = sequence['X_Acceleration'].min()
        max_x_acc = sequence['X_Acceleration'].max()

        mean_y_acc = sequence['Y_Acceleration'].mean()
        std_y_acc = sequence['Y_Acceleration'].std()
        min_y_acc = sequence['Y_Acceleration'].min()
        max_y_acc = sequence['Y_Acceleration'].max()

        mean_acc = sequence['Y_Acceleration'].mean()
        std_acc = sequence['Y_Acceleration'].std()
        min_acc = sequence['Y_Acceleration'].min()
        max_acc = sequence['Y_Acceleration'].max()

        """ Calculate Jerk """
        mean_jerk = sequence['Jerk'].mean()
        std_jerk = sequence['Jerk'].std()
        min_jerk = sequence['Jerk'].min()
        max_jerk = sequence['Jerk'].max()

        """ Calculate angular velocity """
        mean_ang = sequence['Ang_V'].mean()
        std_ang = sequence['Ang_V'].std()
        min_ang = sequence['Ang_V'].min()
        max_ang = sequence['Ang_V'].max()

        """ Calculate Path Tangents """
        mean_tan = sequence['Path_Tangent'].mean()
        std_tan = sequence['Path_Tangent'].std()
        min_tan = sequence['Path_Tangent'].min()
        max_tan = sequence['Path_Tangent'].max()

        # Initialize variables and data structures.
        curve_list = []  # a list to store the curvature values for each segment of the trajectory
        traj_length = 0  # the total length of the trajectory
        accTimeatBeg = 0  # the accumulated time spent in acceleration at the beginning of the trajectory
        numCritPoints = 0  # the number of critical points (where the curvature is very low)
        path = list()  # a list to store the distances traveled for each segment of the trajectory
        flag = True  # a flag to indicate whether the trajectory is in the acceleration phase

        for idx in range(1, 8):
            x_start, x_end = sequence['X'][~sequence['X'].isna()].values[[idx, idx-1]]
            y_start, y_end = sequence['Y'][~sequence['Y'].isna()].values[[idx, idx-1]]
            traj_length += np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
            path.append(traj_length)

            sequence_start, sequence_end = sequence['Timestamp'][~sequence['Timestamp'].isna()].values[[idx, idx-1]]
            dt = sequence_end - sequence_end
            sequence_start, sequence_end = sequence['Speed'][~sequence['Speed'].isna()].values[[idx, idx-1]]
            dv = sequence_end - sequence_end

            if dv > 0 and flag:
                accTimeatBeg += dt
            else:
                flag = False

        for jj in [[mean_x_speed, mean_y_speed, mean_speed, mean_x_acc, mean_y_acc, mean_acc, mean_jerk, mean_ang,
                    mean_curve, mean_tan,
                    std_x_speed, std_y_speed, std_speed, std_x_acc, std_y_acc, std_acc, std_ang, std_jerk,
                    std_curve, std_tan, min_tan,
                    min_x_speed, min_y_speed, min_speed, min_x_acc, min_y_acc, min_acc, min_ang, min_jerk,
                    min_curve,
                    max_x_speed, max_y_speed, max_speed, max_x_acc, max_y_acc, max_acc, max_ang, max_jerk,
                    max_curve, max_tan,
                    elapsed_time, sum_of_angles, accTimeatBeg, traj_length, numCritPoints, button_press_time,
                    mean_speed_over_dist, std_speed_over_dist, min_speed_over_dist, max_speed_over_dist,
                    mean_acceleration_over_dist, std_acceleration_over_dist, max_acceleration_over_dist,
                    min_acceleration_over_dist]]:

    end = time.time()
    took = round(end - start_time, constants.NUM_ROUNDING)
    print(f" > Done! ({took = }s)")



if __name__ == "__main__":
    n = 8

    featured_subjects = []

    preprocessing_start = time.time()
    # Create each individual subjects feature CSV
    for subject in range(constants.SUBJECTS):
        # Translate raw data into a pre-populated dataframe
        prefeatured_dataframe = subject_to_dataframe(f"raw_data/user_{subject}_data.csv")

        # Split the dataframe into sequences of length n.
        sequences = split(prefeatured_dataframe, n)
        print(f"{len(sequences)} sequences of {n} created. (raw shape = {prefeatured_dataframe.shape})")

        # Recalculate entire feature set
        featured_dataframe = process(sequences)

    # utils.join_dataframes_to_csv(featured_subjects, "synth_data/user_all_feature_data.csv")

    preprocessing_end = time.time()
    took = round(preprocessing_end - preprocessing_start, constants.NUM_ROUNDING)
    print(f"Preprocessing complete. ({took = }s)")
