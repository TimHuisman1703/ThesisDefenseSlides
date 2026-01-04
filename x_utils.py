import cv2
import os
import numpy as np

FINAL_FRAMERATE = 60
DEBUG_FRAMERATE = 15

FINAL_SIZE = (1920, 1080)
DEBUG_SIZE = (480, 270)

PATH = os.path.realpath(os.path.dirname(__file__))
PATH_OUTPUT = f"{PATH}\\output"
PATH_FRAME_COUNTS = f"{PATH_OUTPUT}\\frame_counts.csv"

PAUSE_MARKER_COLOR = [86, 52, 18]

C_BLACK = "#000000"
C_DARK_GRAY = "#3F3F3F"
C_GRAY = "#7F7F7F"
C_LIGHT_GRAY = "#BFBFBF"
C_LIGHT_LIGHT_GRAY = "#DFDFDF"
C_WHITE = "#FFFFFF"
C_RED = "#DF2F2F"
C_ORANGE = "#EF7F1F"
C_GREEN = "#0F9F5F"
C_BLUE = "#2F5FBF"
C_PURPLE = "#AF3FBF"

def hex_to_rgb(c):
    c = str(c)
    return np.array([int(c[j:j+2], 16) for j in range(1, 7, 2)])

def rgb_to_hex(r, g, b):
    return ("#" + "".join(hex(round(v))[2:].zfill(2) for v in [r, g, b])).upper()

PAUSE_MARKER_COLOR_HEX = rgb_to_hex(*PAUSE_MARKER_COLOR[::-1])

def lerp(ca, cb, a):
    return rgb_to_hex(*hex_to_rgb(ca) * (1 - a) + hex_to_rgb(cb) * a)

def read_output_videos(verbose=False):
    video_nr = 1
    while (frames := read_output_video(video_nr, verbose=verbose)):
        yield frames
        video_nr += 1

def read_output_video(video_nr, verbose=False):
    path_src_directory = f"{PATH_OUTPUT}/{video_nr:06}"
    if not os.path.exists(path_src_directory):
        return []
    frames = [cv2.imread(f"{path_src_directory}/{filename}") for filename in sorted(os.listdir(path_src_directory))]
    if verbose:
        print(f"\033[30;1mLoaded video #{video_nr} ({len(frames)} frame{'s' * (len(frames) != 1)})\033[0m")
    return frames
