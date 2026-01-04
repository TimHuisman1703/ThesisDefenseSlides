from manim import *
import os
import shutil
import time

from x_utils import *

PATH_RENDER = f"{PATH}\\media\\images\\c_invitation"

def render_invitation():
    start_time = time.time()

    if os.path.exists(PATH_RENDER):
        shutil.rmtree(PATH_RENDER)

    width, height = FINAL_SIZE
    framerate = FINAL_FRAMERATE

    filename = os.path.realpath(__file__)
    command = f"manim {filename} InvitationScene --resolution {width},{height} --frame_rate {framerate} --format=png --disable_caching"

    print(f"\033[0;32m{command}\033[0m")
    exit_code = os.system(command)

    if exit_code != 0:
        return exit_code

    if os.path.exists(PATH_OUTPUT):
        shutil.rmtree(PATH_OUTPUT)
    os.mkdir(PATH_OUTPUT)

    print("\033[34;1mCopying frames...\033[0m")

    for filename in os.listdir(PATH_RENDER):
        nr = int("".join(c for c in filename if c.isdigit()))
        if nr == 0:
            shutil.copyfile(f"{PATH_RENDER}/{filename}", f"{PATH_OUTPUT}/invitation.png")

    duration = int(time.time() - start_time)
    print(f"\033[32;1mFinished in {duration // 60 // 60}h {duration % 60:02}s!\033[0m")

    return exit_code

class InvitationScene(Scene):

    def load_image(self, name):
        image = ImageMobject(f"{PATH}/assets/{name}.png")
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
        return image

    def construct(self):
        background = self.load_image("invitation_background")
        background.set_z_index(0)
        self.add(background)

        all_text_group = Group()
        def add_text(msg, color, width=8, **kwargs):
            text = Text(msg, color=color, **kwargs)
            background = text.copy().set_stroke(C_WHITE, opacity=1, width=width)
            text.set_z_index(10)
            background.set_z_index(10 - 0.1)
            group = Group(background, text)
            all_text_group.add(group)
            return group

        background_fade = Rectangle(C_WHITE, 100, 100).set_fill(C_WHITE, opacity=0.85)
        background_fade.set_z_index(1)
        self.add(background_fade)

        title_text = add_text("A Fast Geometric Multigrid Method\nfor Volumetric Meshes", color=C_BLACK, weight=BOLD) \
            .scale(0.9).move_to(ORIGIN).shift(UP * 1.6)
        invite_text = add_text("Heya! I would like to invite you to my defense, where I'll be bravely defending my work titled", color=C_DARK_GRAY) \
            .scale(0.4).next_to(title_text, UP).shift(DOWN * 0.1)
        dont_worry_text = add_text("Don't worry, I'll bring visuals to\nexplain what these words means...", color=C_GRAY) \
            .scale(0.3).align_to(title_text, DOWN + RIGHT)
        author_text = add_text("MSc Thesis Defense - Tim Huisman", color=C_DARK_GRAY) \
            .scale(0.6).next_to(title_text, DOWN).align_to(title_text, LEFT)
        self.add(author_text)

        when_text = add_text("When?\t  January 23rd (Friday), 10:15 - 12:15", color=C_DARK_GRAY, t2w={"When?": BOLD}, t2c={"When?": C_BLACK}) \
            .scale(0.6).next_to(author_text, DOWN).shift(DOWN * 0.5).align_to(author_text, LEFT)
        when_sub_text_1 = add_text("Presentation in the first 30 minutes, it's possible to leave soon after.", color=C_GRAY) \
            .scale(0.4).next_to(when_text, DOWN).align_to(when_text, LEFT).shift(RIGHT * 2.2 + UP * 0.2)
        when_sub_text_2 = add_text("Please don't be late; we start at 10:15 sharp.", color=C_GRAY) \
            .scale(0.4).next_to(when_sub_text_1, DOWN).align_to(when_text, LEFT).shift(RIGHT * 2.2 + UP * 0.2)
        where_text = add_text("Where?\tBuilding 36, Lecture Hall Chip", color=C_DARK_GRAY, t2w={"Where?": BOLD}, t2c={"Where?": C_BLACK}, t2s={"Chip": ITALIC}) \
            .scale(0.6).next_to(when_sub_text_2, DOWN).align_to(when_text, LEFT)
        where_sub_text = add_text("Mekelweg 4, 2628 CD, Delft", color=C_GRAY) \
            .scale(0.4).next_to(where_text, DOWN).align_to(where_text, LEFT).shift(RIGHT * 2.2 + UP * 0.2)

        online_text = add_text("A link to watch online can be requested closer to the date,", color=C_DARK_GRAY) \
            .scale(0.6).next_to(where_sub_text, DOWN).set_x(0.0).shift(DOWN * 0.2)
        however_text = add_text("but if possible, I would love to see you there in person! :)", color=C_DARK_GRAY) \
            .scale(0.6).next_to(online_text, DOWN).shift(UP * 0.12)

        all_text_group.move_to(ORIGIN)
        self.add(all_text_group)

        eemcs = self.load_image("eemcs").scale_to_fit_height(1.2).next_to(where_text, RIGHT).align_to(where_sub_text, DOWN).shift(np.array([0.2, -0.1, 0.0]))
        eemcs.set_z_index(15)
        self.add(eemcs)

        self.wait(1.5 / FINAL_FRAMERATE)

if __name__ == "__main__":
    render_invitation()
