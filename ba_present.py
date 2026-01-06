import cv2
import gc
import screeninfo
from threading import Thread
import time

from x_utils import *

FINAL = True
FULLSCREEN = True
MOUSE_CONTROLLED = True

LOOK_BACK = 5
LOOK_AHEAD = 15
MARGIN = 5
VERBOSE = True

WINDOW_NAME = "Thesis Defense"

black_screen = np.zeros((1, 1, 3))

loaded_videos = {}
last_frames = {}

def load_video(video_nr):
    loaded_videos[video_nr] = []
    loaded_videos[video_nr] = read_output_video(video_nr, verbose=VERBOSE)

def load_video_range(video_nr, look_back=LOOK_BACK, look_ahead=LOOK_AHEAD, margin=MARGIN):
    indices = [*range(video_nr, video_nr + look_ahead + 1), *range(video_nr - 1, video_nr - look_back, -1)]
    for k in [*loaded_videos.keys()]:
        if k < video_nr - look_back - margin or k > video_nr + look_ahead + margin:
            try:
                del loaded_videos[k]
            except:
                pass

    for idx in indices:
        if idx not in loaded_videos:
            thread = Thread(target=load_video, args=(idx,))
            thread.start()

def load_last_pages():
    global last_frames

    f = open(PATH_FRAME_COUNTS)
    frame_counts = [int(j) for j in f.read().split(",")]
    f.close()

    last_frames.clear()
    for video_nr in range(1, len(frame_counts)):
        path_src = f"{PATH_OUTPUT}/{video_nr:06}/{frame_counts[video_nr] - 1:04}.png"
        if os.path.exists(path_src):
            last_frames[video_nr] = cv2.imread(path_src)

def present(framerate, fullscreen, mouse_controlled):
    global loaded_videos

    num_videos = sum(os.path.isdir(f"{PATH_OUTPUT}/{name}") for name in os.listdir(PATH_OUTPUT))

    thread = Thread(target=load_last_pages)
    thread.start()

    for video_nr in range(1, 4):
        load_video(video_nr)

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
    time_since_video_start = time.time()
    time_since_last_flip = time.time()
    loaded_video_range = False

    frame_nr = -1
    while True:
        if video_nr != video_nr_prev:
            frame_nr_prev = -1
            time_since_last_flip = time.time()
            loaded_video_range = False
            if VERBOSE:
                print(f"\033[34;1mShowing video #{video_nr}\033[0m")
        video_nr_prev = video_nr

        video = loaded_videos.get(video_nr, [])

        frame_nr = max(0, min(int((time.time() - time_since_video_start) * framerate), len(video) - 1))
        frame = video[frame_nr] if video else last_frames.get(video_nr, black_screen)
        if frame_nr_prev != frame_nr:
            cv2.imshow(WINDOW_NAME, frame)
            frame_nr_prev = frame_nr

            if frame_nr == len(video) - 1:
                gc.collect()

        if not loaded_video_range and time.time() - time_since_last_flip > 0.5:
            loaded_video_range = True
            if VERBOSE:
                print("\033[30mLoading...\033[0m")
            load_video_range(video_nr)

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
                time_since_video_start = time.time()
            action = -1
        elif action == 1:
            if video_nr < num_videos:
                video_nr += 1
                time_since_video_start = 0.0
            action = -1
        elif action == 2:
            if video_nr > 1:
                video_nr -= 1
                time_since_video_start = 0.0
            else:
                time_since_video_start = time.time()
            action = -1
        elif action == 3:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    present(
        FINAL_FRAMERATE if FINAL else DEBUG_FRAMERATE,
        FULLSCREEN,
        True
    )
