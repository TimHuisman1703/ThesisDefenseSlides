import cv2
import gc
import screeninfo
from threading import Thread
import time

from x_utils import *

DEBUG = False

WINDOW_NAME = "Thesis Defense"
FULLSCREEN = False
MOUSE_CONTROLLED = True

PRELOAD = 15
POSTLOAD = 5
MARGIN = 5

loaded_videos = {}
last_thread_nr = 0
active_threads = []

def load_video(video_nr):
    loaded_videos[video_nr] = None
    loaded_videos[video_nr] = read_output_video(video_nr, verbose=True)

def load_video_range(video_nr, preload=PRELOAD, postload=POSTLOAD):
    global last_thread_nr

    thread_nr = last_thread_nr + 1
    last_thread_nr = thread_nr

    indices = [*range(video_nr, video_nr + preload + 1), *range(video_nr - 1, video_nr - postload, -1)]
    for k in [*loaded_videos.keys()]:
        if k < POSTLOAD - MARGIN or k > PRELOAD + MARGIN:
            loaded_videos.pop(k)

    for idx in indices:
        if last_thread_nr != thread_nr:
            continue
        if idx not in loaded_videos:
            thread = Thread(target=load_video, args=(idx,))
            thread.start()

def present(framerate, fullscreen, mouse_controlled):
    global loaded_videos

    black_screen = np.zeros((1, 1, 3))
    num_videos = len(os.listdir(OUTPUT_PATH))

    for idx in range(1, 4):
        load_video(idx)

    print(f"\033[34;1mRunning!\033[0m")

    if fullscreen:
        screen = screeninfo.get_monitors()[0]
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(WINDOW_NAME, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        screen = screeninfo.get_monitors()[0]
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(WINDOW_NAME, screen.x - 1, screen.y - 1)
        cv2.resizeWindow(WINDOW_NAME, screen.width // 2, screen.height // 2)

    action = -1
    def click(event, x, y, flags, param):
        nonlocal action

        if event == cv2.EVENT_LBUTTONDOWN:
            action = 0
        elif event == cv2.EVENT_RBUTTONDOWN:
            action = 2

    if mouse_controlled:
        cv2.setMouseCallback(WINDOW_NAME, click)

    video_nr_prev = 0
    video_nr = 1
    time_since_last_click = time.time()

    frame_nr = -1
    while True:
        if video_nr != video_nr_prev:
            frame_nr_prev = -1
            load_video_range(video_nr)

            print(f"\033[34;1mShowing video #{video_nr}\033[0m")
        video_nr_prev = video_nr

        video = loaded_videos.get(video_nr, None)
        if video is None:
            load_video(video_nr)
            load_video_range(video_nr)
            continue

        frame_nr = max(0, min(int((time.time() - time_since_last_click) * framerate), len(video) - 1))
        frame = video[frame_nr] if video else black_screen
        if frame_nr_prev != frame_nr:
            cv2.imshow(WINDOW_NAME, frame)
            frame_nr_prev = frame_nr
        key = cv2.waitKey(1)

        if key in [32, 13]:
            action = 0
        elif key in [100]:
            action = 1
        elif key in [2162688, 2490368, 2424832, 97]:
            action = 2
        elif key in [27]:
            action = 3

        if action == 0:
            if video_nr < num_videos:
                video_nr += 1
                time_since_last_click = time.time()
            action = -1
        elif action == 1:
            if video_nr < num_videos:
                video_nr += 1
                time_since_last_click = 0
            action = -1
        elif action == 2:
            if video_nr > 1:
                video_nr -= 1
                time_since_last_click = 0
            else:
                time_since_last_click = time.time()
            action = -1
        elif action == 3:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    present(
        DEBUG_FRAMERATE if DEBUG else DEFAULT_FRAMERATE,
        FULLSCREEN,
        True
    )
