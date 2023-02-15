import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

    print("Done with test")
