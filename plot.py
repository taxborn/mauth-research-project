import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    # to view a certain subjects data, update this value to the desired subject ID
    subject = 0
    file = f"data/user_{subject}_data.csv"
    # ID,Timestamp,X,Y,Button,Duration
    dataframe = pd.read_csv(file, skiprows=1, names=["ID", "Timestamp", "X", "Y", "Button", "Duration"],
                            usecols=['Timestamp', 'X', 'Y'])

    # We want to iterate over each time stamp, and plot the values
    dataframe.set_index("Timestamp", inplace=True)

    # Split the dataframe into a 2 x n array, and deconstruct those arrays into x's and y's
    x, y = np.hsplit(dataframe, 2)

    # label the plot and set the title
    plt.xlabel('Screen x-coordinates')
    plt.ylabel('Screen y-coordinates')
    plt.title(f"User {subject}'s mouse path")
    # plot the data
    plt.plot(x, y, color='#cba6f7')
    # load the plot
    plt.show()


if __name__ == '__main__':
    main()
