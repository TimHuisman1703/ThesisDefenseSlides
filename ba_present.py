import cv2
import screeninfo
import time

from utils import *

DEBUG = True

WINDOW_NAME = "Thesis Defense"
FULLSCREEN = True
MOUSE_CONTROLLED = True

def present(framerate, fullscreen, mouse_controlled):
    videos = read_output_videos(verbose=True)

    black_screen = np.zeros([video for video in videos if video][0][0].shape)

    print(f"\033[32;1mRunning!\033[0m")

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

    video_nr = 0
    time_since_last_click = time.time()

    while True:
        video = videos[video_nr]
        frame_nr = max(0, min(int((time.time() - time_since_last_click) * framerate), len(video) - 1))
        frame = video[frame_nr] if video else black_screen
        cv2.imshow(WINDOW_NAME, frame)
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
            if video_nr < len(videos) - 1:
                video_nr += 1
                time_since_last_click = time.time()
            action = -1
        elif action == 1:
            if video_nr < len(videos) - 1:
                video_nr += 1
                time_since_last_click = 0
            action = -1
        elif action == 2:
            if video_nr > 0:
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
        not DEBUG or True,
        True
    )
