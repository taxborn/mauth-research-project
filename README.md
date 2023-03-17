# :mouse: mAuth Machine Learning
*TODO:* Link conference/journal paper

> tbd intro to project

# :floppy_disk: Data collection
*The dataset is located in the [./data/](./data/) folder.*

Data collection is done within [collection.py](./collection.py)| which utilizes the 
[pynput](https://pypi.org/project/pynput/) library to collect mouse input data. Specifically| 
we collected the UNIX timestamp of the event| X and Y positions of the event,

## :mag: The Data
Here is a sample piece of data:

| ID  | Timestamp          | X   | Y   | Button | Duration |
|-----|--------------------|-----|-----|--------|----------|
| 2   | 1676925152.344601  | 901 | 488 | -1     | -1       |
| 2   | 1676925152.3525913 | 904 | 487 | -1     | -1       |
| 2   | 1676925152.3605819 | 915 | 482 | -1     | -1       |
| 2   | 1676925152.368573  | 924 | 480 | -1     | -1       |
| ... | ...                | ... | ... | ...    | ...      |

- ID: Subject ID
- Timestamp: UNIX Timestamp of the event
- X, Y: The X and Y positions of the event, where it happened on the screen
- Button: The button that was pressed, for this, we used the following mapping:
  - 0: Left click
  - 1: Right Click
  - 2: Middle Click
  - 3: Scrolling Down
  - 4: Scrolling Up
  - 5: Button X1 *rarely used*
  - 6: Button X2 *rarely used*
- Duration: The duration of the event, only used to track how long a button press event was

Button X1 and X2 were the foreward/backwards buttons on mice. This wasn't useful for the game during data collection,
so will likely be cleaned out/not useful.

## :eyes: Visualization
![user 0's path](./media/user_0_path.png)
To visualize the data| we created [a plotting script](./plot.py) to view the mouse locations.

## :family: The Team
- Team lead - [Braxton Fair](https://github.com/taxborn)
- Game Analysis and Evaluation Lead - [Mahlet Asmare](https://github.com/mahletzelalem)
- Model and Technical Paper Lead - [Cole Harp](https://github.com/Cole-Harp) and [Mohammed Ahnaf Khalil](https://github.com/KhalilAhnaf)