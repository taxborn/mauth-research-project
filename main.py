import os.path
from pynput import mouse
import time

duration = 3  # 60 * 30  # 60 seconds * 30 minutes

user_id = 0  # update for each user
start_position: (int, int) = (0, 0)


def on_move(x, y):
    # On move, log the current position and set the other data cells to -1 to denote an invalid value (we don't need
    # button pressed or duration for just a single move event)
    current_time = time.time()

    with open(os.path.join(data_directory, output_file), "a") as data_file:
        data_file.write(f"\n{user_id},{current_time},{x},{y},-1,-1")
    listener.stop()


def on_click(x, y, button, pressed):
    # TODO: Get press time, is this something we should keep track of in the data here or for ML classifiers?
    # For example, if we keep track of another header, a boolean like was_pressed "pressed" and "released",
    # we can eliminate the need to calculate how long since the last press on release
    if pressed:
        print("Pressed")
        # Check if the last press event was released,
    else:
        print("Released")

    """
    We will denote the button presses as follows:
        0: left click
        1: right click
        2: middle click
        
    There are a few other buttons on my mouse that *might* be of interest, but probably won't be used enough
    to actually be useful
    """
    # Check string representations of the button to determine which button was clicked
    button_map = {
        "Button.left": 0,
        "Button.right": 1,
        "Button.middle": 2,
        "Button.x1": 5,  # Might not use
        "Button.x2": 6,  # Might not use
    }

    button = button_map[str(button)]

    current_time = time.time()

    with open(os.path.join(data_directory, output_file), "a") as data_file:
        data_file.write(f"\n{user_id},{current_time},{x},{y},{button},-1")
    listener.stop()


def on_scroll(x, y, dx, dy):
    """
    We will denote the button presses as follows:
        3: scrolling down event
        4: scrolling up event
    """
    # if dy == 0, we are scrolling up, if dy == -1, we are scrolling down
    is_scrolling_up = dy >= 0
    # if dx == 0 we are scrolling left, if dx == -1 we are scrolling right
    # this is only useful for using a trackpad, so likely won't be used
    is_scrolling_left = dx >= 0

    if is_scrolling_up:
        button = 4
    else:
        button = 3

    current_time = time.time()

    with open(os.path.join(data_directory, output_file), "a") as data_file:
        data_file.write(f"\n{user_id},{current_time},{x},{y},{button},-1")

    listener.stop()


if __name__ == '__main__':
    # Initialize some variables
    output_file = f"user_{user_id}_data_{int(time.time())}.csv"  # adding Unix timestamp to accidental overwrites
    current_directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = f"{current_directory}\data\\"
    was_released = False

    # Try to create the data directory
    try:
        os.mkdir(data_directory)
    except OSError:
        print("Data directory already exists")

    # Check if the output file exists, if it does, wait a bit of time and regenerate the file name
    if os.path.exists(os.path.join(data_directory, output_file)):
        print("duplicate output file found, generating a new file name...")
        # wait 2 seconds to ensure new Unix timestamp
        time.sleep(2)
        output_file = f"user_{user_id}_data_{int(time.time())}.csv"  # adding Unix timestamp to accidental overwrites

    # Add headers to CSV
    # id, Timestamp, X, Y, button, duration,
    #    id - The subject's id
    #    timestamp - the Unix timestmap of the event
    #    X - The x-position of the event
    #    Y - The y-position of the event
    #    duration - The duration of the event, for example, how long the subject held down the left mouse button
    with open(os.path.join(data_directory, output_file), "w") as file:
        file.write("ID,Timestamp,X,Y,Button,Duration")

    # Create a delay to give time to switch to the game
    delay = 3
    print(f"{delay} seconds until mouse events are recorded...")
    time.sleep(delay)
    # Set the starting position
    start_position = mouse.Controller().position

    start = time.time()
    end = start + duration
    print(f"{start = } {end = }")

    # Start listening for events. Main loop
    while time.time() < end:
        with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
            listener.join()
    listener.stop()

    print("Completed!")
