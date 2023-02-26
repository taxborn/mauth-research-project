import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # ID,Timestamp,X,Y,Button,Duration
    user = 0
    file = f"data/user_{user}_data.csv"
    test_df = pd.read_csv(file, skiprows=1, names=["ID", "Timestamp", "X", "Y", "Button", "Duration"],
                          usecols=['Timestamp', 'X', 'Y'])

    test_df.set_index("Timestamp", inplace=True)
    plt.rcParams.update({'font.size': 12})
    test_df = test_df.to_numpy()

    x, y = np.hsplit(test_df, 2)

    plt.xlabel('Screen x-coordinates')
    plt.ylabel('Screen y-coordinates')
    plt.title(f"User {user}'s mouse path")
    plt.plot(x, y, color='#30B262')
    plt.show()

    print("Done with test")
