import os.path

from pynput import mouse
import time

global output_file
user_id = 1  # update for each user
start_position: (int, int) = (0, 0)  # TODO


def on_move(x, y):
    print(f"Mouse moved to ({x}, {y})")


def on_click(x, y, button, pressed):
    if pressed:
        print("Pressed at: ", end=" ")
    else:
        print("Released at: ", end=" ")

    print(f"({x}, {y}), {button = }")


def on_scroll(x, y, dx, dy):
    # if dx == -1, we are scrolling to the right
    # if dx == 0, we are scrolling to the right
    if dy < 0:
        print("Scrolled down at: ", end=" ")
    else:
        print("Scrolled up at: ", end=" ")

    print(f"({x}, {y}), {dx = } {dy = }")


def main():
    global output_file
    output_file = f"user_{user_id}_data_{int(time.time())}.csv"  # adding Unix timestamp to accidental overwrites
    current_directory = os.path.dirname(os.path.realpath(__file__))
    data_directory = current_directory + "\data\\"

    print(f"current dir:{current_directory} data: {data_directory}")

    try:
        os.mkdir(data_directory)
    except OSError as error:
        print("Data directory already exists")

    # Check if the output file exists, if it does, wait a bit of time and regenerate the file name
    if os.path.exists(os.path.join(data_directory, output_file)):
        print("duplicate output file found, generating a new file name...")
        # wait 2 seconds to ensure new Unix timestamp
        time.sleep(2)
        output_file = f"user_{user_id}_data_{int(time.time())}.csv"  # adding Unix timestamp to accidental overwrites


    # Add headers to CSV
    with open(os.path.join(data_directory, output_file), "w") as file:
        file.write("test test")

    # Start listening for events. Main loop
    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()


if __name__ == '__main__':
    main()
