import numpy as np
import pandas as pd
from collections import deque
import os.path


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
    df.fillna(0)
    print(f"Size: {df.size} \nShape {df.shape} \nColumn Names: {df.columns}")
    df = sequence_maker(df)
    return df


def sequence_maker(df, sequence_length=8):
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
            sharp_angles = np.sum(abs(temp[1:, 12]) < .0005)

            for jj in [[mean_x_speed, mean_y_speed, mean_speed, mean_x_acc, mean_y_acc, mean_acc, mean_jerk, mean_ang,
                        mean_curve, mean_tan,
                        std_x_speed, std_y_speed, std_speed, std_x_acc, std_y_acc, std_acc, std_ang, std_jerk,
                        std_curve, std_tan, min_tan,
                        min_x_speed, min_y_speed, min_speed, min_x_acc, min_y_acc, min_acc, min_ang, min_jerk,
                        min_curve,
                        max_x_speed, max_y_speed, max_speed, max_x_acc, max_y_acc, max_acc, max_ang, max_jerk,
                        max_curve, max_tan,
                        elapsed_time, sum_of_angles, accTimeatBeg, traj_length, numCritPoints, button_press_time]]:
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
                               'button_press_time'])
    df.insert(0, 'ID', ID)
    print(f"Head: {df.head()} \nSize: {df.size} \nShape {df.shape} \nColumn Names: {df.columns}")
    df.to_csv(f"extracted_features_data/user_{ID}_extracted_{sequence_length}.csv", index=False)
    return df


if __name__ == "__main__":
    for i in range(15):
        subj_df = data_to_df(f"data/user_{i}_data.csv")
        print('done')
