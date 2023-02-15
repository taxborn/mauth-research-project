import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque

SEQ_LEN = 60
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates


# def preprocess(df):
#     for col in df.columns:
#         if col != "Subject ID":
#             df[col] = preprocessing.normalize([i[:-1] for i in df.values], axis=1)
#     df = df.values
#     sequential_data = []
#     prev_data = deque(maxlen=SEQ_LEN)
#     for i in df:
#         prev_data.append([n for n in i[:-1]])
#         if len(prev_data) == SEQ_LEN:
#             sequential_data.append([np.array(prev_data), i[-1]])
#     random.shuffle(sequential_data)
#     X = []
#     y = []
#     for seq, target in sequential_data:
#         X.append(seq)
#         y.append(target)
#     return np.array(X), np.array(y)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # ID,Timestamp,X,Y,Button,Duration
    file = "data/user_99_data_1676439130.csv"
    test_df = pd.read_csv(file, skiprows=1, names=["ID", "Timestamp", "X", "Y", "Button", "Duration"],
                          usecols=['Timestamp', 'X', 'Y'])

    test_df.set_index("Timestamp", inplace=True)
    plt.rcParams.update({'font.size': 12})
    test_df = test_df.to_numpy()

    x, y = np.hsplit(test_df, 2)

    plt.plot(x, y, color='#FF0000')
    plt.xlabel('Screen x-coordinates')
    plt.ylabel('Screen y-coordinates')
    plt.title('Mouse path')
    plt.plot(x, y, color='#67F8A0')
    plt.show()

    # test_X, test_y = preprocess(test_df)
    print("Done with test")
