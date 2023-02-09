from pynput import mouse
import os


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
    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()


if __name__ == '__main__':
    main()
