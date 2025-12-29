import cv2
import os
import numpy as np

DEFAULT_FRAMERATE = 60
DEBUG_FRAMERATE = 15

DEFAULT_SIZE = (1920, 1080)
DEBUG_SIZE = (480, 270)

PATH = os.path.realpath(os.path.dirname(__file__))
OUTPUT_PATH = f"{PATH}\\output"

PAUSE_MARKER_COLOR = [86, 52, 18]

C_BLACK = "#000000"
C_DARK_GRAY = "#3F3F3F"
C_GRAY = "#7F7F7F"
C_LIGHT_GRAY = "#BFBFBF"
C_WHITE = "#FFFFFF"
C_RED = "#DF2F2F"
C_ORANGE = "#EF7F1F"
C_GREEN = "#0F9F5F"
C_BLUE = "#2F5FBF"
C_PURPLE = "#AF3FBF"

def hex_to_rgb(c):
    return np.array([int(c[j:j+2], 16) for j in range(1, 7, 2)])

def rgb_to_hex(r, g, b):
    return f"#{hex(round(r))[2:]:02}{hex(round(g))[2:]:02}{hex(round(b))[2:]:02}".upper()

PAUSE_MARKER_COLOR_HEX = rgb_to_hex(*PAUSE_MARKER_COLOR[::-1])

def lerp(ca, cb, a):
    return rgb_to_hex(*hex_to_rgb(ca) * (1 - a) + hex_to_rgb(cb) * a)

def read_output_videos(verbose):
    frame_filenames = sorted(f"{OUTPUT_PATH}/{filename}" for filename in sorted(os.listdir(OUTPUT_PATH)))

    videos = [[]]
    for frame_filename in frame_filenames:
        frame = cv2.imread(frame_filename)
        if all(list(frame[np.random.randint(0, frame.shape[0]), np.random.randint(0, frame.shape[1])]) == PAUSE_MARKER_COLOR for _ in range(100)):
            if verbose:
                print(f"\033[30;1mLoaded video #{len(videos)} ({len(videos[-1])} frame{'s' * (len(videos[-1]) != 1)})\033[0m")
            videos.append([])
        else:
            videos[-1].append(frame)

    if videos[-1]:
        print(f"\033[30;1mLoaded video #{len(videos)} ({len(videos[-1])} frame{'s' * (len(videos[-1]) != 1)})\033[0m")
    while not videos[-1]:
        videos.pop()

    return videos
