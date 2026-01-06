import heapq
from manim import *
import numpy as np
import os
import scipy.sparse as sp
import shutil
import time

from x_utils import *

FINAL = True
FROM = 0
TO = 10000
CONVERT_AFTER = False
PRESENT_AFTER = True

BACKGROUND_COLOR = C_WHITE
SIZE = (8.0 * 16 / 9, 8.0)

PATH_RENDER = f"{PATH}\\media\\images\\a_render"

config.background_color = BACKGROUND_COLOR
config.max_files_cached = 1000

def render_slides():
    start_time = time.time()

    if os.path.exists(PATH_RENDER):
        shutil.rmtree(PATH_RENDER)

    width, height = FINAL_SIZE if FINAL else DEBUG_SIZE
    framerate =  FINAL_FRAMERATE if FINAL else DEBUG_FRAMERATE

    filename = os.path.realpath(__file__)
    command = f"manim {filename} PresentationScene --resolution {width},{height} --frame_rate {framerate} --format=png --disable_caching --from_animation_number {FROM},{TO}"

    print(f"\033[0;32m{command}\033[0m")
    exit_code = os.system(command)

    if exit_code != 0:
        return exit_code

    if os.path.exists(PATH_OUTPUT):
        shutil.rmtree(PATH_OUTPUT)
    os.mkdir(PATH_OUTPUT)

    print("\033[34;1mWriting frames...\033[0m")

    filenames = os.listdir(PATH_RENDER)
    filenames_indexed = []
    for filename in filenames:
        nr = int("".join(c for c in filename if c.isdigit()))
        filenames_indexed.append((nr, filename))
    filenames_indexed = sorted(filenames_indexed)
    filenames = [pair[1] for pair in sorted(filenames_indexed)]

    video_nr = 1
    frame_nr = 0
    frame_counts = {}
    for filename in filenames:
        nr = int("".join(c for c in filename if c.isdigit()))
        if nr % 100 == 0:
            print(f"\033[30;1m  {nr}/{len(filenames)}")

        path_src = f"{PATH_RENDER}/{filename}"
        frame = cv2.imread(path_src)
        if all(list(frame[np.random.randint(0, frame.shape[0]), np.random.randint(0, frame.shape[1])]) == PAUSE_MARKER_COLOR for _ in range(100)):
            video_nr += 1
            frame_nr = 0
            continue

        path_dst_directory = f"{PATH_OUTPUT}/{video_nr:06}"
        if not os.path.exists(path_dst_directory):
            os.mkdir(path_dst_directory)

        path_dst = f"{path_dst_directory}/{frame_nr:04}.png"
        shutil.copyfile(path_src, path_dst)
        frame_counts[video_nr] = frame_counts.get(video_nr, 0) + 1
        frame_nr += 1
    print(f"\033[30;1m  {len(filenames)}/{len(filenames)}\033[0m")

    f = open(PATH_FRAME_COUNTS, "w")
    f.write(",".join(str(frame_counts.get(idx, 0)) for idx in range(max([0, *frame_counts]) + 1)))
    f.close()

    duration = int(time.time() - start_time)
    print(f"\033[32;1mFinished in {duration // 60 // 60}h {duration // 60 % 60:02}m {duration % 60:02}s!\033[0m")

    return exit_code

class PresentationScene(MovingCameraScene):
    ########################################
    #                                      #
    #            HELPER METHODS            #
    #                                      #
    ########################################

    def all_objects(self):
        to_exclude = {self.page_number_text, self.page_number_background}
        return [obj for obj in filter(lambda x: issubclass(type(x), Mobject), self.mobjects) if obj not in to_exclude]

    def pause(self, text=None):
        self.hold(0.15)

        pause_marker_rectangle = Rectangle(PAUSE_MARKER_COLOR_HEX, 100, 100).set_fill(PAUSE_MARKER_COLOR_HEX, opacity=1)
        pause_marker_rectangle.set_z_index(10 ** 10)
        self.add_foreground_mobject(pause_marker_rectangle)
        self.add(pause_marker_rectangle)
        self.wait(1)

        self.update_page()

        self.remove(pause_marker_rectangle)

        nr = self.renderer.num_plays - 2
        color = [30, 34, 36][(FROM <= nr <= TO) + (nr == FROM)]
        print(f"\033[{color};1m> [{self.page_number - 1:>4} : {nr:<4}]" + f" {text}" * (text is not None) + "\033[0m")

    def hold(self, run_time):
        ignore_me = Dot().move_to(UP * 100)
        self.add(ignore_me)
        self.play(
            ignore_me.animate.shift(UP * 100),
            run_time=run_time
        )
        self.remove(ignore_me)

    def flatten(self, *mobjects):
        flattened = []
        for mobject in mobjects:
            if type(mobject) == Group:
                flattened.extend(self.flatten(*mobject))
            else:
                flattened.append(mobject)
        return flattened

    def into_frame(self, *mobjects, invert=False):
        scale = self.camera.frame_height / SIZE[1]
        shift = self.camera.frame_center

        for mobject in self.flatten(*mobjects):
            if not invert:
                mobject \
                    .scale(scale, about_point=ORIGIN) \
                    .shift(shift) \
                    .set_stroke(width=mobject.get_stroke_width() * scale)
            else:
                mobject \
                    .shift(-shift) \
                    .scale(1 / scale, about_point=ORIGIN) \
                    .set_stroke(width=mobject.get_stroke_width() / scale)

    def save_state(self, *mobjects):
        for mobject in self.flatten(*mobjects):
            mobject.state_center = mobject.get_center()
            mobject.state_width = mobject.width
            mobject.state_height = mobject.height
            mobject.state_stroke_width = mobject.get_stroke_width()
        return mobjects

    def load_state(self, *mobjects):
        for mobject in self.flatten(*mobjects):
            mobject.move_to(mobject.state_center)
            mobject.stretch_to_fit_width(mobject.state_width)
            mobject.stretch_to_fit_height(mobject.state_height)
            mobject.set_stroke(width=mobject.state_stroke_width)
        return mobjects

    def fix(self, *mobjects: Mobject, absolute=False):
        for mobject in self.flatten(*mobjects):
            if not absolute:
                self.into_frame(mobject, invert=True)
            self.save_state(mobject)
            mobject.add_updater(self.fix_updater)

    def unfix(self, *mobjects: Mobject):
        for mobject in self.flatten(*mobjects):
            mobject.remove_updater(self.fix_updater)

    def create_title(self, title):
        title_tex = Tex(f"\\underline{{\\textbf{{{title}}}}}", color=C_BLACK)
        title_tex.set_z_index(92)
        title_background = title_tex.copy().set_stroke(C_WHITE, width=16)
        title_background.set_z_index(91)
        title_group = Group(title_tex, title_background)
        title_group.scale(1.1).to_corner(UP + LEFT)

        return title_group

    def update_page(self):
        if self.page_number_text:
            self.remove(self.page_number_text)
            self.remove(self.page_number_background)

        self.page_number += 1

        self.page_number_background = Rectangle(C_WHITE, 0.4, 0.6).round_corners(0.1).set_fill(opacity=1).set_stroke(C_WHITE, opacity=0.5)
        self.page_number_background.to_corner(DOWN + RIGHT)
        self.page_number_background.set_z_index(101)
        self.add(self.page_number_background)

        self.page_number_text = Text(str(self.page_number), color=C_DARK_GRAY).set_stroke(opacity=0).scale(0.4)
        self.page_number_text.move_to(self.page_number_background)
        self.page_number_text.set_z_index(102)
        self.add(self.page_number_text)

        self.fix(self.page_number_background, self.page_number_text, absolute=True)

    def load_image(self, name):
        image = ImageMobject(f"{PATH}/assets/{name}.png")
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
        return image

    def clear(self, run_time=0.0):
        if run_time:
            white_rectangle = Rectangle(WHITE, 100, 100).set_fill(opacity=1)
            white_rectangle.set_z_index(99)
            self.play(
                FadeIn(white_rectangle),
                run_time=run_time
            )

        self.remove(*self.all_objects())
        self.camera.frame.move_to(ORIGIN)
        self.camera.frame.scale_to_fit_height(SIZE[1])

    def create_arrow(self, arrow, opacity=1):
        start, end = arrow.get_start_and_end()
        opacity = arrow.get_stroke_opacity()
        arrow.set_opacity(0).put_start_and_end_on(start, start + 1e-5 * (end - start))
        return arrow.animate.set_opacity(opacity).put_start_and_end_on(start, end)

    def pseudo_uniform_points(self, nr=32, dist=0.5, iterations=100, seed=0):
        np.random.seed(seed)
        ps = np.random.uniform(-1.0, 1.0, (nr, 2))

        for _ in range(iterations):
            nps = ps.copy()
            for i in range(nr):
                diffs = []
                for j in range(nr):
                    if i != j:
                        diff = ps[i] - ps[j]
                        norm = np.linalg.norm(diff)
                        if norm > 0 and norm < dist:
                            diffs.append(diff / norm ** 2)
                direction = np.zeros((2,)) + np.sum(diffs, axis=0)
                direction *= 0.01
                nps[i] = np.array([
                    min(max(nps[i][c] + direction[c], -1.0), 1.0)
                    for c in range(2)
                ])
            ps = nps

        return ps

    def create_code_diagram(self, *stages: str, z_index=0):
        pos_prv = ORIGIN
        pos = ORIGIN

        group = Group()
        for stage in stages:
            if not stage:
                pos += DOWN * 0.2
                continue

            rect = Rectangle(C_LIGHT_GRAY, 0.6, 3.0).set_stroke(width=4).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5), opacity=1).round_corners(0.1)
            rect.move_to(pos)
            rect.set_z_index(z_index + 0.6)
            rect_back = rect.copy().set_stroke(C_WHITE, width=rect.get_stroke_width() + 8)
            rect_back.set_z_index(z_index + 0.2)

            text = Text(stage, color=C_BLACK).scale(0.55)
            text.move_to(pos)
            text.set_z_index(z_index + 0.8)

            lines = []
            if np.linalg.norm(pos - pos_prv) > 1e-3:
                line = Line(pos_prv, pos).set_stroke(C_GRAY, width=12).set_cap_style(CapStyleType.ROUND)
                line.set_z_index(z_index + 0.4)
                line_back = line.copy().set_stroke(C_WHITE, width=line.get_stroke_width() + 8)
                line_back.set_z_index(z_index)
                lines.append(Group(line, line_back))

            group.add(
                Group(
                    Group(
                        Group(rect, rect_back),
                        text,
                    ),
                    *lines
                )
            )

            pos_prv, pos = pos, pos + DOWN * 0.7

        return group

    def construct(self):
        self.fix_updater = lambda obj: self.into_frame(*self.load_state(obj))

        self.page_number = 0
        self.page_number_text = None
        self.update_page()

        self.animate()

    ################################
    #                              #
    #            SLIDES            #
    #                              #
    ################################

    def animate_slide_first_page(self):
        background = self.load_image("invitation_background")
        background.set_z_index(0)
        self.add(background)

        def add_text(msg, color, width=8, **kwargs):
            text = Text(msg, color=color, **kwargs)
            background = text.copy().set_stroke(C_WHITE, opacity=1, width=width)
            text.set_z_index(10)
            background.set_z_index(10 - 0.1)
            group = Group(background, text)
            return group

        background_fade = Rectangle(C_WHITE, 100, 100).set_fill(C_WHITE, opacity=0.92)
        background_fade.set_z_index(1)
        self.add(background_fade)

        title_text = add_text("A Fast Geometric Multigrid Method\nfor Volumetric Meshes", color=C_BLACK, weight=BOLD) \
            .scale(0.9).move_to(ORIGIN).shift(UP * 1.6)
        author_text = add_text("MSc Thesis Defense - Tim Huisman", color=C_DARK_GRAY) \
            .scale(0.6).next_to(title_text, DOWN).align_to(title_text, LEFT)
        self.add(author_text)

        spot_tet_mg = [None] * 3
        for i in range(3):
            spot_tet_mg[i] = self.load_image(f"spot_tet_mg_{i}").scale(0.7)
            spot_tet_mg[i].move_to(DOWN * 2.5 + (i - 1) * (RIGHT * 3.0 + DOWN * 0.4))
            spot_tet_mg[i].set_z_index(5)

        group = Group(
            title_text,
            author_text,
            *spot_tet_mg
        )
        group.move_to(DOWN * 0.2)
        self.add(group)

        self.pause("First page")

        self.clear()

    def animate_slide_problem_summary(self):
        pass

    def animate_slide_dirichlet_demonstration(self):
        self.pause("Start dirichlet_demonstration")

        title_tex = self.create_title("Example")
        self.play(
            FadeIn(title_tex),
            run_time=0.6
        )
        self.fix(title_tex)
        self.pause("Show title")

        n = 16
        size = 6

        def val_to_deg(v):
            return 15 + 15 * v

        A_rows, A_cols, A_vals = [], [], []
        b_vals = [0] * (n + 2) ** 2
        for y in range(n + 2):
            for x in range(n + 2):
                idx = y * (n + 2) + x

                fixed = None
                if x == 0 or y == n + 1:
                    fixed = 1
                if y == 0:
                    fixed = 0
                if x == n + 1:
                    fixed = -1

                if fixed is not None:
                    A_rows.append(idx)
                    A_cols.append(idx)
                    A_vals.append(1.0)
                    b_vals[idx] = fixed
                else:
                    A_rows.extend([idx, idx, idx, idx, idx])
                    A_cols.extend([idx, idx - 1, idx + 1, idx - (n + 2), idx + (n + 2)])
                    A_vals.extend([4.0, -1.0, -1.0, -1.0, -1.0])
                    b_vals[idx] = 0.0

        A = sp.csc_array((A_vals, (A_rows, A_cols)), dtype=np.float32)
        b = np.array(b_vals)
        values = sp.linalg.inv(A) @ b

        temperatures = [[0] * n for _ in range(n)]
        for iy in range(n):
            for ix in range(n):
                temperatures[iy][ix] = values[(iy + 1) * (n + 2) + (ix + 1)]

        spacing = size / n
        margin = 0.01

        walls = [None] * 4
        a, b = spacing * n / 2 + 0.01, spacing * (n / 2 + 1) - 0.01
        walls[0] = Polygon(
            np.array([-a, a, 0]),
            np.array([-b, b, 0]),
            np.array([-b, -b, 0]),
            np.array([-a, -a, 0]),
        ).set_fill(C_ORANGE, opacity=1).set_stroke(opacity=0)
        walls[1] = walls[0].copy().rotate(np.pi / 2, about_point=ORIGIN).set_fill(C_LIGHT_GRAY)
        walls[2] = walls[1].copy().rotate(np.pi / 2, about_point=ORIGIN).set_fill(C_BLUE)
        walls[3] = walls[2].copy().rotate(np.pi / 2, about_point=ORIGIN).set_fill(C_ORANGE)
        self.play(
            FadeIn(walls[0], shift=0.5 * RIGHT),
            FadeIn(walls[1], shift=0.5 * UP),
            FadeIn(walls[2], shift=0.5 * LEFT),
            FadeIn(walls[3], shift=0.5 * DOWN),
            run_time=0.8
        )
        self.pause("Setup walls")

        cells = [[None] * n for _ in range(n)]
        cells_hot = [[None] * n for _ in range(n)]
        cells_cold = [[None] * n for _ in range(n)]
        question_marks = [[None] * n for _ in range(n)]
        for iy in range(n):
            for ix in range(n):
                x = size * ((ix + 0.5) / n - 0.5)
                y = size * ((iy + 0.5) / n - 0.5)

                temperature = temperatures[iy][ix]

                cell = Square(spacing - margin).round_corners(0.05).set_stroke(width=0).set_fill(C_LIGHT_GRAY, opacity=1)
                cell.move_to((x, y, 0))
                cell.set_z_index(2)
                cells[iy][ix] = cell

                cell_hot = Square(spacing - margin).round_corners(0.05).set_stroke(width=0).set_fill(C_ORANGE, opacity=1).set_opacity(temperature)
                cell_hot.move_to((x, y, 0))
                cell_hot.set_z_index(3)
                cells_hot[iy][ix] = cell_hot

                cell_cold = Square(spacing - margin).round_corners(0.05).set_stroke(width=0).set_fill(C_BLUE, opacity=1).set_opacity(-temperature)
                cell_cold.move_to((x, y, 0))
                cell_cold.set_z_index(3)
                cells_cold[iy][ix] = cell_cold

                question_mark = Tex("$\\textbf{?}$", color=C_DARK_GRAY)
                question_mark.scale(0.8).move_to(cell)
                question_mark.set_z_index(3)
                question_marks[iy][ix] = question_mark
        cells_flat = [obj for row in cells for obj in row if obj]
        cells_hot_flat = [obj for row in cells_hot for obj in row if obj]
        cells_cold_flat = [obj for row in cells_cold for obj in row if obj]
        question_marks_flat = [obj for row in question_marks for obj in row if obj]

        indices_examples = [3 * n + (n - 3), (n - 2) * n + 5]
        for idx in indices_examples:
            self.play(
                FadeIn(cells_flat[idx], scale=5),
                FadeIn(question_marks_flat[idx], scale=5),
                run_time=0.4
            )
            self.pause(f"Introduce node {idx}")

            self.play(
                FadeOut(question_marks_flat[idx], scale=0),
                FadeIn(cells_hot_flat[idx], scale=0),
                FadeIn(cells_cold_flat[idx], scale=0),
                run_time=0.4
            )
            self.pause(f"Solve node {idx}")

        self.play(
            *[FadeIn(obj, scale=0) for j, obj in enumerate(cells_flat) if j not in indices_examples],
            *[FadeIn(obj, scale=0) for j, obj in enumerate(question_marks_flat) if j not in indices_examples],
            run_time=0.4
        )
        self.pause("Show all unsolved positions")

        animations = []
        indices = sorted(range(len(cells_flat)), key=lambda x: (x % n) + -(x // n))
        for idx in indices:
            if idx in indices_examples:
                continue
            animations.append(
                AnimationGroup(
                    FadeOut(question_marks_flat[idx], scale=0),
                    FadeIn(cells_hot_flat[idx], scale=0),
                    FadeIn(cells_cold_flat[idx], scale=0),
                    run_time=0.4
                )
            )
        self.play(
            AnimationGroup(
                *animations,
                lag_ratio=0.01,
                run_time=1.6
            ),
            run_time=1.6
        )
        self.pause("Solve them all")

        fx, fy = 2, 4

        to_zoom_disappear = Group(*cells_hot_flat, *cells_cold_flat, *[obj for j, obj in enumerate(cells_flat) if j != fy * n + fx])
        camera_zoom_1 = 3.6
        camera_shift_1 = cells[fy][fx].get_center() + LEFT * (1.5 * spacing)
        self.play(
            *[FadeOut(obj) for obj in to_zoom_disappear],
            FadeIn(question_marks[fy][fx], scale=0),
            self.camera.frame.animate.scale(1 / camera_zoom_1).shift(camera_shift_1),
            run_time=1.6
        )
        self.pause("Zoom in")

        temps = [1.0, 0.6, 0.6, 0.8]
        temperature_texts = [None] * 5
        fade_in_animations = []
        for d in range(4):
            nx = fx - (d == 0) + (d == 1)
            ny = fy - (d == 2) + (d == 3)

            temperature = temps[d]
            cells[ny][nx].set_opacity(opacity=1)
            cells_hot[ny][nx].set_opacity(opacity=temperature)
            cells_cold[ny][nx].set_opacity(opacity=-temperature)

            temperature_texts[d] = MathTex(f"{val_to_deg(temperature):.1f}\\small{{^{{\\circ}}\\text{{C}}}}").scale(0.2)
            temperature_texts[d].move_to(cells[ny][nx])
            temperature_texts[d].set_z_index(12)

            fade_in_animations.extend([
                FadeIn(cells[ny][nx], scale=0),
                FadeIn(cells_hot[ny][nx], scale=0),
                FadeIn(cells_cold[ny][nx], scale=0),
                FadeIn(temperature_texts[d], scale=0)
            ])

        self.play(
            *fade_in_animations,
            run_time=0.6
        )
        self.pause("Show neighbors")

        temperature_avg = sum(temps) / 4
        temperature_texts[4] = MathTex(f"{val_to_deg(temperature_avg):.1f}\\small{{^{{\\circ}}\\text{{C}}}}").scale(0.2)
        temperature_texts[4].move_to(cells[fy][fx])
        temperature_texts[4].set_z_index(12)

        cells_hot[fy][fx].set_opacity(temperature_avg)
        cells_cold[fy][fx].set_opacity(-temperature_avg)
        to_move = []
        to_keep_in_front = []
        for d in range(4):
            nx = fx - (d == 0) + (d == 1)
            ny = fy - (d == 2) + (d == 3)

            to_keep_in_front.append(cells[ny][nx])
            to_keep_in_front.append(cells_hot[ny][nx])
            to_keep_in_front.append(cells_cold[ny][nx])
            to_keep_in_front.append(temperature_texts[d])

            for cs in [cells_hot, cells_cold]:
                c_copy = cs[ny][nx].copy()
                c_copy.generate_target()
                c_copy.target.move_to(cells[fy][fx])
                c_copy.target.set_opacity(0)
                to_move.append(c_copy)

        for obj in to_keep_in_front:
            obj.set_z_index(obj.get_z_index() + 50)
        self.play(
            *[MoveToTarget(obj) for obj in to_move],
            FadeIn(cells_hot[fy][fx]),
            FadeIn(cells_cold[fy][fx]),
            FadeOut(question_marks[fy][fx], scale=0),
            FadeIn(temperature_texts[4], scale=0),
            run_time=0.8
        )
        for obj in to_keep_in_front:
            obj.set_z_index(obj.get_z_index() - 50)
        self.remove(*to_move)
        self.pause("Take average heat")

        cell_outlines = [None] * 5
        for d in range(5):
            nx = fx - (d == 0) + (d == 1)
            ny = fy - (d == 2) + (d == 3)

            cell_outline = cells[ny][nx].copy()
            cell_outline.set_stroke(color=C_PURPLE if d == 4 else C_GREEN, opacity=1, width=2.4).set_fill(opacity=0)
            cell_outline.set_z_index(10 + (d == 4))
            cell_outlines[d] = cell_outline
        self.play(
            Create(cell_outlines[4]),
            run_time=0.4
        )
        self.pause("Show middle outline")

        self.play(
            *[Create(obj) for obj in cell_outlines[:4]],
            run_time=0.4
        )
        self.pause("Show neighbor outlines")

        system_container = Rectangle(C_GRAY, 1.05, 0.8).set_stroke(width=1.2).set_fill(C_WHITE, opacity=1)
        system_container.move_to(camera_shift_1).shift((-1.2, 0.0, 0.0))
        system_container.set_z_index(20)
        equation_tex = MathTex("x", "={", "y", "+", "y", "+", "y", "+", "y", "\\over 4}").set_fill(C_BLACK).scale(0.16)
        equation_tex.set_color_by_tex("x", C_WHITE)
        equation_tex.set_color_by_tex("y", C_WHITE)
        equation_tex.move_to(system_container.get_top() + np.array([0, -0.12, 0]))
        equation_tex.set_z_index(25)
        tex_parts = equation_tex.get_parts_by_tex("y")
        tex_parts.add(equation_tex.get_part_by_tex("x"))
        move_animations = []
        equation_cells = [None] * 5
        for d in range(5):
            cell_equation = cell_outlines[d].copy()
            cell_equation.set_z_index(30)
            move_animations.append(
                cell_equation.animate
                    .set_stroke(width=1.2)
                    .set_fill(LIGHT_GRAY, opacity=1)
                    .scale(0.24 if d == 4 else 0.12)
                    .move_to(tex_parts[d])
                    .shift(np.array([-0.02, 0.006, 0.0]) if d == 4 else np.array([0.0, 0.01, 0.0]))
            )
            equation_cells[d] = cell_equation

        self.play(
            FadeIn(system_container, scale=0.8),
            *move_animations,
            run_time=1.2
        )
        self.hold(0.2)
        self.play(
            Write(equation_tex),
            run_time=0.8
        )
        self.pause("Write first equation")

        self.play(
            FadeOut(temperature_texts[4], scale=0),
            FadeOut(cells_hot[fy][fx]),
            FadeIn(question_marks[fy][fx], scale=0),
            run_time=0.4
        )
        self.pause("Make middle unknown again")

        animations = []
        for d in range(4):
            nx = fx - (d == 0) + (d == 1)
            ny = fy - (d == 2) + (d == 3)

            animations.extend([
                FadeOut(temperature_texts[d], scale=0),
                FadeOut(cells_hot[ny][nx]),
                FadeIn(question_marks[ny][nx], scale=0),
            ])
        self.play(
            *animations,
            run_time=0.4
        )
        self.pause("Make neighbors unknown too")

        animations = []
        for iy in range(n):
            for ix in range(n):
                if abs(fx - ix) + abs(fy - iy) == 2:
                    animations.extend([
                        FadeIn(cells[iy][ix], scale=0),
                        FadeIn(question_marks[iy][ix], scale=0),
                    ])
        self.play(
            *animations,
            run_time=0.4
        )
        self.pause("Add two-ring vertices")

        equations_tex = [equation_tex] + [equation_tex.copy() for _ in range(4)]
        equations_cells = [equation_cells] + [[obj.copy() for obj in equation_cells] for _ in range(4)]
        for i in range(1, 5):
            offset = DOWN * 0.2 * i
            equations_tex[i].generate_target()
            equations_tex[i].target.shift(offset)
            for j in range(5):
                equations_cells[i][j].generate_target()
                equations_cells[i][j].target.shift(offset)
                if j == 4:
                    equations_cells[i][j].target.set_stroke(C_GREEN)
                elif j == i - 1:
                    equations_cells[i][j].target.set_stroke(C_PURPLE)
                else:
                    equations_cells[i][j].target.set_stroke(opacity=0)
        self.play(
            *[MoveToTarget(equations_cells[i][j]) for i in range(1, 5) for j in range(5)],
            *[MoveToTarget(equations_tex[i]) for i in range(1, 5)],
            run_time=0.8
        )
        self.pause("Add next four equations")

        animations = []
        for iy in range(n):
            for ix in range(n):
                if abs(fx - ix) + abs(fy - iy) == 3:
                    animations.extend([
                        FadeIn(cells[iy][ix], scale=0),
                        FadeIn(question_marks[iy][ix], scale=0),
                    ])
        cells_boundary = [[None] * n for _ in range(4)]
        cells_boundary_flat = []
        for i in range(4):
            for j in range(n):
                jx, jy = [(j, 0), (j, n - 1), (0, j), (n - 1, j)][i]
                cell = cells[jy][jx].copy()
                cell.set_stroke(C_WHITE, width=2, opacity=1).set_fill([C_LIGHT_GRAY, C_ORANGE, C_ORANGE, C_BLUE][i], opacity=1)
                cell.shift([DOWN, UP, LEFT, RIGHT][i] * spacing)
                cells_boundary[i][j] = cell
                cells_boundary_flat.append(cell)
        animations.extend([
            FadeIn(cells_boundary[2][fy], scale=0),
        ])
        self.play(
            *animations,
            run_time=0.4
        )
        self.pause("Add three-ring vertices")

        base = 4
        amount = 48 if FINAL else 8
        equations_tex += [equations_tex[base].copy() for _ in range(amount)]
        equations_cells += [[obj.copy() for obj in equations_cells[base]] for _ in range(amount)]
        equation_cell_orange = None
        for i in range(5, 5 + amount):
            offset = DOWN * 0.2 * (i - base)
            equations_tex[i].generate_target()
            equations_tex[i].target.shift(offset)
            for j in range(5):
                equations_cells[i][j].generate_target()
                equations_cells[i][j].target.shift(offset)
                if (i, j) == (6, 2):
                    equations_cells[i][j].target.set_fill(C_ORANGE)
                    equation_cell_orange = equations_cells[i][j]
                if (i, j) in [(5, 3), (6, 0), (7, 1), (7, 3), (9, 1), (10, 2)]:
                    equations_cells[i][j].target.set_stroke(C_GREEN, opacity=1)
                else:
                    equations_cells[i][j].target.set_stroke(opacity=0)
        factor = 8
        self.play(
            *[MoveToTarget(equations_cells[i][j]) for i in range(5, len(equations_cells)) for j in range(5)],
            *[MoveToTarget(equations_tex[i]) for i in range(5, len(equations_tex))],
            system_container.animate.stretch(factor, dim=1).shift(DOWN * system_container.height * (factor - 1) / 2),
            run_time=0.8
        )
        self.pause("Add many more equations")

        camera_shift_2 = LEFT * 1.8

        system_group = Group(system_container, *equations_tex)
        for row in equations_cells:
            system_group.add(*row)
        for obj in system_group:
            obj.generate_target()
            obj.target.shift(-camera_shift_1).scale(camera_zoom_1, about_point=ORIGIN).shift(camera_shift_2)
            if obj == system_container:
                obj.target.set_stroke(width = obj.get_stroke_width() * camera_zoom_1)
            else:
                obj.target.set_stroke(opacity=0)
        equation_cell_orange.target.set_fill(C_LIGHT_GRAY)
        animations = []
        for iy in range(n):
            for ix in range(n):
                if abs(fx - ix) + abs(fy - iy) > 3:
                    animations.extend([
                        FadeIn(cells[iy][ix], scale=0),
                        FadeIn(question_marks[iy][ix], scale=0),
                    ])
        self.play(
            *[FadeOut(obj) for obj in cell_outlines],
            self.camera.frame.animate.shift(-camera_shift_1).scale(camera_zoom_1).shift(camera_shift_2),
            *[MoveToTarget(obj) for obj in system_group],
            run_time=0.6
        )
        self.play(
            *animations,
            FadeOut(cells_boundary[2][fy]),
            run_time=0.6
        )
        self.pause("Put camera back, make all cells appear")

        self.play(
            system_group.animate.shift(DOWN * 0.3),
            run_time=0.6
        )
        matrix_equation_tex = Tex("$A\\mathbf{x} = \\mathbf{b}$", color=C_BLACK).scale(1.2)
        matrix_equation_tex.next_to(system_group, UP)
        system_group.add(matrix_equation_tex)
        self.play(
            Write(matrix_equation_tex),
            run_time=0.4
        )
        self.pause("Show matrix equation")

        # TODO: consider direct solve?

        title_tex_old = title_tex
        self.unfix(title_tex_old)

        title_tex = self.create_title("Iterative Solving: Gauss-Seidel")
        self.into_frame(title_tex)
        self.play(
            FadeOut(title_tex_old),
            FadeIn(title_tex, shift=DOWN),
            run_time=0.6
        )
        title_tex.to_corner(UP + LEFT)
        self.fix(title_tex, absolute=True)
        self.pause("Replace title")

        np.random.seed(2)
        states = [
            np.array([[np.random.rand() * 2 - 1 for _ in range(n + 2)] for _ in range(n + 2)])
        ]
        iterations_to_show = [
            (1, 1, 0.8, smooth),
            (2, 1, 0.6, smooth),
            (5, 1, 0.25, lambda _: 1),
            (20, 1 if FINAL else 2, 0.04 if FINAL else 0.08, lambda _: 1),
            (50, 1 if FINAL else 4, 0.02 if FINAL else 0.08, lambda _: 1)
        ]
        iterations = max([j[0] for j in iterations_to_show])
        for obj in cells_cold_flat + cells_hot_flat:
            obj.set_opacity(0)
            self.add(obj)
        for i in range(n):
            states[0][-1][i] = 0.0
            states[0][n][i] = 1.0
            states[0][i][-1] = 1.0
            states[0][i][n] = -1.0
        for it in range(iterations):
            state = states[-1].copy()
            for iy in range(n):
                for ix in range(n):
                    state[iy][ix] = 0.25 * sum(state[iy - (d == 0) + (d == 1)][ix - (d == 2) + (d == 3)] for d in range(4))
            states.append(state)

        animations = []
        for iy in range(n):
            for ix in range(n):
                animations.extend([
                    cells_hot[iy][ix].animate.set_fill(opacity=states[0][iy][ix]),
                    cells_cold[iy][ix].animate.set_fill(opacity=-states[0][iy][ix]),
                ])
        self.play(
            *[FadeOut(obj, scale=0) for obj in question_marks_flat],
            *animations,
            run_time=0.6
        )
        self.pause("Initialize randomly")

        camera_zoom_3 = 6
        camera_shift_3 = cells[-1][0].get_center()
        self.play(
            self.camera.frame.animate.shift(-camera_shift_2).scale(1 / camera_zoom_3).shift(camera_shift_3),
            FadeIn(cells_boundary[1][0]),
            FadeIn(cells_boundary[2][-1]),
            run_time=1.6
        )
        self.pause("Zoom into top left")
        self.remove(system_group)

        amount = 6
        for idx in range(amount):
            fx, fy = idx, n - 1

            animations = []
            to_move = []
            to_keep_in_front = []
            for d in range(4):
                nx = fx - (d == 0) + (d == 1)
                ny = fy - (d == 2) + (d == 3)

                objs = []
                if ny >= n:
                    objs.append(cells_boundary[1][nx])
                elif nx < 0:
                    objs.append(cells_boundary[2][ny])
                else:
                    objs.append(cells[ny][nx])
                    objs.append(cells_hot[ny][nx])
                    objs.append(cells_cold[ny][nx])
                for obj in objs:
                    to_keep_in_front.append(obj)
                    obj_moving = obj.copy()
                    obj_moving.set_stroke(opacity=0)
                    obj_moving.generate_target()
                    obj_moving.target.move_to(cells[fy][fx])
                    obj_moving.target.set_opacity(0)
                    to_move.append(obj_moving)

            for obj in to_keep_in_front:
                obj.set_z_index(obj.get_z_index() + 50)
            self.play(
                *[MoveToTarget(obj) for obj in to_move],
                cells_hot[fy][fx].animate.set_opacity(states[1][fy][fx]),
                cells_cold[fy][fx].animate.set_opacity(-states[1][fy][fx]),
                run_time=0.8 if idx < 2 else 0.6
            )
            for obj in to_keep_in_front:
                obj.set_z_index(obj.get_z_index() - 50)
            self.remove(*to_move)
            if idx < 2:
                self.pause(f"Perform first GS at cell {idx}")

            if idx == amount - 1:
                continue

            self.play(
                *[FadeOut(cells_boundary[2][-1])] * (idx == 0),
                FadeOut(cells_boundary[1][idx]),
                FadeIn(cells_boundary[1][idx + 1]),
                self.camera.frame.animate.shift(RIGHT * spacing),
                run_time=0.8 if idx < 2 else 0.6
            )
            if idx < 1:
                self.pause(f"Go to cell {idx}")
        self.pause(f"Do {amount} GS calculations")

        indices = []
        for iy in range(n - 1, -1, -1):
            for ix in range(n):
                indices.append((ix, iy))
        indices = indices[amount:]
        first_gs_animations = []
        for jx, jy in indices:
            first_gs_animations.append(cells_hot[jy][jx].animate.set_opacity(states[1][jy][jx]))
            first_gs_animations.append(cells_cold[jy][jx].animate.set_opacity(-states[1][jy][jx]))
        first_gs_animation = AnimationGroup(
            *first_gs_animations,
            lag_ratio=0.01
        )

        camera_shift_4 = UP * 0.24
        self.play(
            first_gs_animation,
            self.camera.frame.animate.shift((amount - 1) * LEFT * spacing).shift(-camera_shift_3).scale(camera_zoom_3).shift(camera_shift_4),
            FadeOut(cells_boundary[1][amount - 1]),
            run_time=1.6
        )
        self.pause("Put camera back again")

        iteration_counter_text = None
        it = 0
        for goal, step_size, run_time, func in iterations_to_show:
            while it < goal:
                it = min(goal, it + step_size)

                iteration_counter_text_new = MathTex(f"{it}", "\\text{ iteration}", "\\text{s}", color=C_DARK_GRAY)
                if goal == 1:
                    iteration_counter_text_new[-1].set_color(C_WHITE)
                iteration_counter_text_new.scale(0.8).next_to(walls[3], UP).shift(DOWN * 0.04)
                text_animations = []
                if it == 1:
                    text_animations.append(FadeIn(iteration_counter_text_new, scale=0.8))
                else:
                    text_animations.append(TransformMatchingTex(iteration_counter_text, iteration_counter_text_new))

                color_animations = [
                    cells_hot[jy][jx].animate.set_opacity(states[it][jy][jx])
                    for jy in range(n) for jx in range(n)
                ] + [
                    cells_cold[jy][jx].animate.set_opacity(-states[it][jy][jx])
                    for jy in range(n) for jx in range(n)
                ]

                self.play(
                    *text_animations,
                    *color_animations,
                    run_time=run_time,
                    rate_func=func
                )

                iteration_counter_text = iteration_counter_text_new

            self.pause(f"Show state {it}")

        walls_big = [obj.copy().set_fill(opacity=0) for obj in walls]

        cells_big = [[None] * (n // 2) for _ in range(n // 2)]
        cells_big_flat = []
        cells_hot_big = [[None] * (n // 2) for _ in range(n // 2)]
        cells_hot_big_flat = []
        cells_cold_big = [[None] * (n // 2) for _ in range(n // 2)]
        cells_cold_big_flat = []
        for iy in range(n // 2):
            for ix in range(n // 2):
                x = size * (2 * (ix + 0.5) / n - 0.5)
                y = size * (2 * (iy + 0.5) / n - 0.5)

                cell_big = Square(2 * spacing - margin).round_corners(0.05).set_stroke(width=0).set_fill(C_LIGHT_GRAY, opacity=0)
                cell_big.move_to((x, y, 0))
                cells_big[iy][ix] = cell_big
                cells_big_flat.append(cell_big)

                cell_hot = cell_big.copy().set_fill(C_ORANGE, opacity=0)
                cell_hot.set_z_index(cell_hot.get_z_index() + 1)
                cells_hot_big[iy][ix] = cell_hot
                cells_hot_big_flat.append(cell_hot)

                cell_cold = cell_big.copy().set_fill(C_BLUE, opacity=0)
                cell_cold.set_z_index(cell_cold.get_z_index() + 1)
                cells_cold_big[iy][ix] = cell_cold
                cells_cold_big_flat.append(cell_cold)

        np.random.seed(0)
        states_big = [
            np.array([[np.random.rand() * 2 - 1 for _ in range(n // 2 + 2)] for _ in range(n // 2 + 2)])
        ]
        iterations_big = 10
        for i in range(n // 2):
            states_big[0][-1][i] = 0.0
            states_big[0][n // 2][i] = 1.0
            states_big[0][i][-1] = 1.0
            states_big[0][i][n // 2] = -1.0
        for it in range(iterations_big):
            state_big = states_big[-1].copy()
            for iy in range(n // 2):
                for ix in range(n // 2):
                    state_big[iy][ix] = 0.25 * sum(state_big[iy - (d == 0) + (d == 1)][ix - (d == 2) + (d == 3)] for d in range(4))
            states_big.append(state_big)

        camera_shift_5 = np.array([4.4, 0.4, 0.0])
        camera_zoom_5 = 0.8

        animations = []
        for iy in range(n // 2):
            for ix in range(n // 2):
                cells_hot_big[iy][ix].set_fill(opacity=states_big[0][iy][ix])
                animations.append(
                    cells_hot_big[iy][ix].animate.set_fill(opacity=states_big[0][iy][ix]).shift((2 * camera_shift_5[0]) * RIGHT)
                )

                cells_cold_big[iy][ix].set_fill(opacity=-states_big[0][iy][ix])
                animations.append(
                    cells_cold_big[iy][ix].animate.set_fill(opacity=-states_big[0][iy][ix]).shift((2 * camera_shift_5[0]) * RIGHT)
                )

        self.play(
            *[obj.animate.set_fill(opacity=1).shift((2 * camera_shift_5[0]) * RIGHT) for obj in cells_big_flat],
            *animations,
            *[obj.animate.set_fill(opacity=1).shift((2 * camera_shift_5[0]) * RIGHT) for obj in walls_big],
            self.camera.frame.animate.scale(1 / camera_zoom_5).shift(camera_shift_5),
            run_time=1.2
        )
        self.pause("Show bigger grid")

        iteration_counter_big_text = None
        indices = sorted(range(len(cells_big_flat)), key=lambda x: (x % (n // 2)) + -(x // (n // 2)))
        for it in range(1, iterations_big + 1):
            for idx in indices:
                ix, iy = idx % (n // 2), idx // (n // 2)
                cells_hot_big_flat[idx].set_fill(opacity=states_big[it][iy][ix])
                cells_cold_big_flat[idx].set_fill(opacity=-states_big[it][iy][ix])

            if iteration_counter_big_text is not None:
                self.remove(iteration_counter_big_text)
            iteration_counter_big_text_new = MathTex(f"{it}", "\\text{ iteration}", "\\text{s}", color=C_DARK_GRAY)
            if it == 1:
                iteration_counter_big_text_new[-1].set_color(C_WHITE)
            iteration_counter_big_text_new.scale(0.8).next_to(walls_big[3], UP).shift(DOWN * 0.04)
            iteration_counter_big_text = iteration_counter_big_text_new
            self.add(iteration_counter_big_text)

            self.hold(0.15)
        self.pause(f"Relax the big grid for {iterations_big} iterations")

        plus_rectangle_1 = Rectangle(C_DARK_GRAY, 1.2, 0.2).set_stroke(opacity=0).set_fill(opacity=1).shift(camera_shift_5)
        plus_rectangle_2 = Rectangle(C_DARK_GRAY, 0.2, 1.2).set_stroke(opacity=0).set_fill(opacity=1).move_to(plus_rectangle_1)
        plus_question_tex = MathTex("\\textbf{?}", color=C_DARK_GRAY).scale(2).next_to(plus_rectangle_1, UP)
        self.play(
            FadeIn(plus_rectangle_1, scale=0.5),
            FadeIn(plus_rectangle_2, scale=0.5),
            FadeIn(plus_question_tex, scale=0.5, target_position=camera_shift_5),
            run_time=0.6
        )
        self.pause("Make plus with question mark appear")

        white_fade = Rectangle(C_WHITE, 100, 100).set_fill(opacity=0.75)
        white_fade.set_z_index(50)
        multigrid_text = Text("\"Multigrid\"?", color=C_BLACK, t2c={"Multi": C_RED}).scale(2.4).move_to(camera_shift_5)
        multigrid_text.set_z_index(52)
        multigrid_text_background = multigrid_text.copy().set_stroke(C_WHITE, width=8)
        multigrid_text_background.set_z_index(51)
        self.play(
            FadeIn(white_fade),
            FadeIn(multigrid_text, shift=UP * 0.5),
            FadeIn(multigrid_text_background, shift=UP * 0.5),
            run_time=0.8
        )
        self.pause("Show \"Multigrid\" text")

        self.clear(run_time=0.6)

    def animate_slide_multigrid_diagram(self):
        title_tex = self.create_title("Multigrid")

        icons = []

        size = 0.8
        margin = 0.02
        gap = 0.16

        spacing_vert = 2.0

        n = 4
        for m in [n, n // 2, n // 4]:
            spacing = size / m
            icon = Group()
            for iy in range(m):
                for ix in range(m):
                    cell = Square(spacing - margin).round_corners(0.05).set_fill(C_GRAY, opacity=1).set_stroke(opacity=0)
                    cell.shift(np.array([ix * spacing, iy * spacing, 0]))
                    icon.add(cell)
            icons.append(icon)

        for i in range(3):
            icons[i].move_to(np.array([-5.6, spacing_vert * (1 - i), 0.0]))

        timeline_0 = Line(LEFT * 4.8, RIGHT * 20.0).set_stroke(C_LIGHT_GRAY, width=8).set_cap_style(CapStyleType.ROUND).set_y(icons[0].get_y())
        timeline_0.set_z_index(1)

        self.play(
            FadeIn(title_tex),
            FadeIn(icons[0], scale=1.5),
            run_time=0.6
        )
        self.fix(title_tex)
        self.play(
            Create(timeline_0),
            rate_func=rush_into,
            run_time=0.8
        )
        self.pause("Draw initial timeline")

        gs_icon_square = Square(0.44).set_stroke(C_GRAY, width=4).set_fill(lerp(C_WHITE, C_ORANGE, 0.5), opacity=1).round_corners(0.1)
        gs_icon_letters = Text("GS", color=C_DARK_GRAY).scale(0.4)
        gs_icon_template = Group(gs_icon_square, gs_icon_letters)
        gs_icon_template.set_z_index(20)

        solve_icon_square = Rectangle(C_GRAY, 0.44, 0.84).set_stroke(C_GRAY, width=4).set_fill(lerp(C_WHITE, C_GREEN, 0.5), opacity=1).round_corners(0.1)
        solve_icon_letters = Text("Solve", color=C_DARK_GRAY).scale(0.4)
        solve_icon_template = Group(solve_icon_square, solve_icon_letters)
        solve_icon_template.set_z_index(20)

        gs_number = 21
        gs_icons = []
        for i in range(gs_number):
            gs_icon = gs_icon_template.copy()
            gs_icon.shift(UP * spacing_vert + RIGHT * (i * (gs_icon.get_width() + gap) - 4.4))
            gs_icons.append(gs_icon)

        self.play(
            AnimationGroup(
                *[FadeIn(obj, scale=3.0) for obj in gs_icons],
                lag_ratio=0.05
            ),
            run_time=0.8
        )
        self.pause("Show GS's")

        self.play(
            *[obj[0].animate.set_stroke(C_RED).set_fill(lerp(C_WHITE, C_RED, 0.5)).scale(0.8) for obj in gs_icons[2:]],
            *[obj[1].animate.set_fill(C_RED).scale(0.8) for obj in gs_icons[2:]],
            run_time=0.4
        )
        self.hold(0.8)
        self.play(
            *[FadeOut(obj) for obj in gs_icons[2:]],
            run_time=0.6
        )
        self.pause("Remove unnecessary GS's")

        gap_vert = 0.36
        res_length = 0.4
        start = gs_icons[1].get_right()[0] + gap
        conn_line_0_a = Line(
            np.array([start, spacing_vert, 0.0]),
            np.array([start, spacing_vert - gap_vert, 0.0])
        ).set_stroke(C_LIGHT_LIGHT_GRAY, width=8).set_cap_style(CapStyleType.ROUND)
        res_line_0_a = Line(
            conn_line_0_a.get_end(),
            np.array([start + res_length, spacing_vert - gap_vert, 0.0]),
        ).set_stroke(C_LIGHT_GRAY, width=8).set_cap_style(CapStyleType.ROUND)
        res_line_0_a.set_z_index(1)
        subtract_tex = Tex("$\\mathbf{-}$", color=C_DARK_GRAY).scale(0.5).move_to(np.array([res_line_0_a.get_x(), conn_line_0_a.get_y(), 0.0]))
        self.play(
            Create(conn_line_0_a),
            run_time=0.2,
            rate_func=rush_into
        )
        self.play(
            Create(res_line_0_a),
            FadeIn(subtract_tex, scale=0),
            run_time=0.2,
            rate_func=rush_from
        )
        self.pause("Introduce residual equation 0")

        arrow_x_travel = 0.6
        restrict_01_arrow = Arrow(stroke_width=8).put_start_and_end_on(
            res_line_0_a.get_end(),
            np.array([res_line_0_a.get_end()[0] + arrow_x_travel, 0.0, 0.0])
        ).set_color(C_PURPLE).set_cap_style(CapStyleType.ROUND)
        restrict_01_arrow.set_z_index(11)

        solve_1 = solve_icon_template.copy()
        solve_1.move_to(restrict_01_arrow.get_end() + RIGHT * (solve_1.get_width() / 2 + gap))
        timeline_1 = timeline_0.copy()
        timeline_1.put_start_and_end_on(
            restrict_01_arrow.get_end(),
            restrict_01_arrow.get_end() + RIGHT * (2 * gap + solve_1.get_width())
        )
        angle = np.arctan2(*(restrict_01_arrow.get_end() - restrict_01_arrow.get_start())[:2][::-1])
        restrict_01_text = Text("Restrict", color=C_PURPLE).scale(0.5).rotate(angle)
        restrict_01_text.move_to(restrict_01_arrow).shift(LEFT * 0.36).shift((restrict_01_arrow.get_end() - restrict_01_arrow.get_start()) * 0.02)

        self.play(
            FadeIn(icons[1], scale=1.5),
            run_time=0.6,
        )
        self.pause("Introduce mid domain")

        self.play(
            self.create_arrow(restrict_01_arrow),
            run_time=0.6,
        )
        self.play(
            Create(timeline_1),
            run_time=0.4,
            rate_func=rush_from
        )
        self.pause("Draw first restriction")

        self.play(
            Write(restrict_01_text),
            run_time=0.4
        )
        self.pause("Draw restriction text")

        self.play(
            FadeIn(solve_1, scale=3.0),
            run_time=0.6
        )
        self.pause("Show solve 1")

        start = solve_1.get_right()[0] + gap
        res_line_0_b = res_line_0_a.copy()
        res_line_0_b.shift(RIGHT * (res_length + arrow_x_travel * 2 + timeline_1.get_width()))
        prolong_10_arrow = restrict_01_arrow.copy().put_start_and_end_on(
            timeline_1.get_end(),
            res_line_0_b.get_start()
        )
        prolong_10_text = Text("Prolong", color=C_PURPLE).scale(0.5).rotate(-angle)
        prolong_10_text.move_to(prolong_10_arrow).shift(RIGHT * 0.28).shift((prolong_10_arrow.get_end() - prolong_10_arrow.get_start()) * -0.1)
        self.play(
            self.create_arrow(prolong_10_arrow),
            run_time=0.6
        )
        self.play(
            Create(res_line_0_b),
            run_time=0.3,
            rate_func=rush_from
        )
        self.hold(0.2)
        self.play(
            Write(prolong_10_text),
            run_time=0.4
        )
        self.pause("Draw first prolongation")

        conn_line_0_b = Line(
            res_line_0_b.get_end(),
            res_line_0_b.get_end() + gap_vert * UP
        ).set_stroke(C_LIGHT_LIGHT_GRAY, width=8).set_cap_style(CapStyleType.ROUND)
        add_tex = Tex("$\\mathbf{+}$", color=C_DARK_GRAY).scale(0.5).move_to(np.array([res_line_0_b.get_x(), conn_line_0_b.get_y(), 0.0]))
        self.play(
            Create(conn_line_0_b),
            FadeIn(add_tex, scale=0.0),
            run_time=0.3
        )
        self.pause("Add error correction")

        gs_icons = gs_icons[:2] + [gs_icon_template.copy() for _ in range(2)]
        gs_icons[2].move_to(res_line_0_b.get_end() + RIGHT * (gs_icons[2].get_width() / 2 + gap)).set_y(spacing_vert)
        gs_icons[3].move_to(gs_icons[2].get_center() + RIGHT * (gs_icons[2].get_width() + gap))
        self.play(
            AnimationGroup(
                *[FadeIn(obj, scale=3.0) for obj in gs_icons[2:]],
                lag_ratio=0.05
            ),
            run_time=0.8
        )
        self.pause("Show post-GS's")

        self.play(
            solve_1.animate.scale(1.6),
            run_time=0.3
        )
        self.hold(0.8)
        self.play(
            solve_1.animate.scale(1 / 1.6),
            run_time=0.3
        )
        self.pause("Highlight mid solve")

        to_recurse = [
            *gs_icons,
            conn_line_0_a,
            res_line_0_a,
            subtract_tex,
            res_line_0_b,
            conn_line_0_b,
            add_tex,
            restrict_01_arrow,
            restrict_01_text,
            prolong_10_arrow,
            prolong_10_text,
            timeline_1,
            solve_1
        ]
        to_move_right = [
            prolong_10_arrow,
            prolong_10_text,
            res_line_0_b,
            conn_line_0_b,
            add_tex,
            gs_icons[2],
            gs_icons[3]
        ]
        offset_right_down = solve_1.get_left() - gs_icons[0].get_left()
        for idx, obj in enumerate(to_recurse):
            obj = obj.copy()
            to_recurse[idx] = obj
            obj.shift(offset_right_down)
        restrict_12_arrow = to_recurse[-6]
        restrict_12_text = to_recurse[-5]
        prolong_21_arrow = to_recurse[-4]
        prolong_21_text = to_recurse[-3]
        offset_right = RIGHT * (gs_icons[3].get_left()[0] + gs_icons[3].get_width() - gs_icons[0].get_left()[0] - solve_1.get_width())
        self.play(
            *[obj.animate.shift(offset_right) for obj in to_move_right],
            timeline_1.animate.put_start_and_end_on(
                timeline_1.get_start(),
                timeline_1.get_end() + RIGHT * offset_right
            ),
            solve_1.animate.shift(offset_right / 2),
            run_time=0.8
        )
        to_recurse_group = Group(*to_recurse)
        self.play(
            solve_1[0].animate
                .set_fill(opacity=0)
                .set_stroke(opacity=0)
                .move_to(to_recurse_group)
                .set_width(to_recurse_group.get_width() + 0.2)
                .set_height(to_recurse_group.get_height() + 0.2),
            solve_1[1].animate
                .set_fill(opacity=0)
                .move_to(to_recurse_group),
            FadeIn(to_recurse_group, scale=0, target_position=solve_1.get_center()),
            FadeIn(icons[2], scale=0.8),
            run_time=0.8
        )
        self.remove(solve_1)
        self.pause("Recurse solve")

        diff = 0.28
        solve_2 = to_recurse[-1]
        solve_rectangle_new = Rectangle(C_GRAY, 0.44 + diff, 0.84).set_stroke(C_GRAY, width=4).set_fill(lerp(C_WHITE, C_GREEN, 0.5), opacity=1).round_corners(0.1)
        solve_rectangle_new.move_to(solve_2)
        solve_rectangle_new.set_z_index(solve_2[0].get_z_index())
        direct_text = Text("Direct", color=C_DARK_GRAY).scale(0.4).set_fill(opacity=0)
        direct_text.move_to(solve_2)
        direct_text.set_z_index(solve_2[1].get_z_index())
        self.play(
            Transform(solve_2[0], solve_rectangle_new, replace_mobject_with_target_in_scene=True),
            solve_2[1].animate.shift(DOWN * (diff / 2)),
            direct_text.animate.shift(UP * (diff / 2)).set_fill(opacity=1),
            run_time=0.6
        )
        self.pause("Show direct solve")

        v_cycle_text = Text("V-Cycle", color=C_DARK_GRAY).scale(1.0)
        v_cycle_text.move_to(solve_2).shift(DOWN * 1.2)
        self.play(
            Write(v_cycle_text),
            run_time=0.6
        )
        self.pause("Show V-Cycle text")

        v_line_1 = Line().set_stroke(C_BLUE, width=12).set_cap_style(CapStyleType.ROUND)
        start = (gs_icons[0].get_center() + gs_icons[1].get_center()) * 0.5
        end = solve_2.get_center()
        v_line_1.put_start_and_end_on(start - 0.1 * (end - start), end)
        v_line_1.set_z_index(50)
        start = solve_2.get_center()
        end = (gs_icons[2].get_center() + gs_icons[3].get_center()) * 0.5
        v_line_2 = v_line_1.copy().put_start_and_end_on(start, end - 0.1 * (start - end))
        v_line_2.set_z_index(50)
        self.play(
            v_cycle_text[0].animate.set_color(C_BLUE),
            Create(v_line_1),
            run_time=0.3,
            rate_func=rush_into
        )
        self.play(
            Create(v_line_2),
            run_time=0.3,
            rate_func=rush_from
        )
        self.pause("Show V-structure")

        camera_shift_1 = RIGHT * 4.0
        camera_zoom_1 = 0.7
        self.play(
            self.camera.frame.animate.scale(1 / camera_zoom_1).shift(camera_shift_1),
            run_time=0.6
        )

        repeats = 8
        to_fade_out = []
        v_shape = Group(v_line_1, v_line_2)
        for it in range(repeats):
            v_shape = v_shape.copy()
            v_shape.generate_target()
            if it == 0:
                v_shape.target.stretch(0.25, dim=0)
            v_shape.target.shift(0.5 * (v_shape.target.get_width() + v_shape.get_width()) * RIGHT)
            to_fade_out.append(v_shape)
            self.play(
                MoveToTarget(v_shape),
                run_time=0.4
            )
        self.pause("Repeat V's")

        self.play(
            self.camera.frame.animate.shift(-camera_shift_1).scale(camera_zoom_1),
            v_cycle_text[0].animate.set_color(C_DARK_GRAY),
            FadeOut(v_line_1),
            FadeOut(v_line_2),
            *[FadeOut(obj) for obj in to_fade_out],
            run_time=0.6
        )
        self.pause("Put camera back")

        self.play(
            *[obj.animate.set_fill(C_BLUE).scale(1.6, about_point=icons[0].get_center()) for obj in icons[0]],
            run_time=0.3
        )
        self.hold(0.8)
        self.play(
            *[obj.animate.scale(1 / 1.6, about_point=icons[0].get_center()) for obj in icons[0]],
            run_time=0.3
        )
        self.pause("Highlight fine domain")

        highlight_domain_1 = Square(1.2).set_stroke(C_RED, width=12, opacity=0.75).round_corners(0.2).set_cap_style(CapStyleType.ROUND)
        highlight_domain_1.set_z_index(25)
        highlight_domain_1.move_to(icons[1])
        highlight_domain_2 = highlight_domain_1.copy()
        highlight_domain_2.move_to(icons[2])
        self.play(
            Create(highlight_domain_1),
            Create(highlight_domain_2),
            *[obj.animate.set_fill(C_RED) for obj in icons[1]],
            *[obj.animate.set_fill(C_RED) for obj in icons[2]],
            run_time=0.6
        )
        self.pause("Highlight missing domains")

        highlight_restrict_01 = Rectangle(C_RED, 2.16, 1.28).set_stroke(C_RED, width=12, opacity=0.75).round_corners(0.2).set_cap_style(CapStyleType.ROUND)
        highlight_restrict_01.set_z_index(25)
        highlight_restrict_01.move_to(Group(restrict_01_arrow, restrict_01_text))
        highlight_restrict_12 = highlight_restrict_01.copy()
        highlight_restrict_12.shift(offset_right_down)
        highlight_prolong_10 = highlight_restrict_01.copy()
        highlight_prolong_10.shift(RIGHT * (Group(prolong_10_arrow, prolong_10_text).get_x() - Group(restrict_01_arrow, restrict_01_text).get_x()))
        highlight_prolong_21 = highlight_prolong_10.copy()
        highlight_prolong_21.shift(np.array([-offset_right_down[0], offset_right_down[1], 0.0]))
        self.play(
            Create(highlight_restrict_01),
            Create(highlight_restrict_12),
            Create(highlight_prolong_21),
            Create(highlight_prolong_10),
            restrict_01_arrow.animate.set_color(C_RED),
            restrict_01_text.animate.set_color(C_RED),
            prolong_10_arrow.animate.set_color(C_RED),
            prolong_10_text.animate.set_color(C_RED),
            restrict_12_arrow.animate.set_color(C_RED),
            restrict_12_text.animate.set_color(C_RED),
            prolong_21_arrow.animate.set_color(C_RED),
            prolong_21_text.animate.set_color(C_RED),
            run_time=0.6
        )
        self.pause("Highlight missing transfer operators")

        self.clear(run_time=0.6)

    def animate_slide_prolongation_demonstration(self):
        n = 8
        size = 2.4
        margin = 0.04

        dims = [n, n // 2, n]
        states = [np.zeros((dims[j], dims[j]), dtype=float) for j in range(3)]
        np.random.seed(0)
        for iy in range(n):
            for ix in range(n):
                x, y = (ix + 0.5) / n, (iy + 0.5) / n
                states[0][iy][ix] = x ** 2 - y ** 2 + np.random.normal(0.0, 0.5)
        for iy in range(n // 2):
            for ix in range(n // 2):
                states[1][iy][ix] = sum(
                    states[0][jy][jx]
                    for jy in range(2 * iy, 2 * (iy + 1))
                    for jx in range(2 * ix, 2 * (ix + 1))
                ) / 4
        for iy in range(n):
            for ix in range(n):
                states[2][iy][ix] = states[1][iy // 2][ix // 2]

        cells = [[[None] * d for _ in range(d)] for d in dims]
        cells_hot = [[[None] * d for _ in range(d)] for d in dims]
        cells_cold = [[[None] * d for _ in range(d)] for d in dims]
        for i in range(3):
            spacing = size / dims[i]

            for iy in range(dims[i]):
                for ix in range(dims[i]):
                    x = size * ((ix + 0.5) / dims[i] - 0.5)
                    y = size * ((iy + 0.5) / dims[i] - 0.5)

                    temperature = states[i][iy][ix]

                    cell = Square((1 - margin) * spacing).round_corners(0.05).set_stroke(width=0).set_fill(C_LIGHT_GRAY, opacity=1)
                    cell.move_to((x, y, 0))
                    cells[i][iy][ix] = cell

                    cell_hot = Square((1 - margin) * spacing).round_corners(0.05).set_stroke(width=0).set_fill(C_ORANGE, opacity=1).set_opacity(temperature)
                    cell_hot.move_to((x, y, 0))
                    cell_hot.set_z_index(1)
                    cells_hot[i][iy][ix] = cell_hot

                    cell_cold = Square((1 - margin) * spacing).round_corners(0.05).set_stroke(width=0).set_fill(C_BLUE, opacity=1).set_opacity(-temperature)
                    cell_cold.move_to((x, y, 0))
                    cell_cold.set_z_index(1)
                    cells_cold[i][iy][ix] = cell_cold

        px, py = 2.0, 2.2
        centers = [
            np.array([-px, py, 0.0]),
            np.array([0.0, -py, 0.0]),
            np.array([px, py, 0.0]),
        ]

        cells_flat = [[obj for row in c for obj in row if obj] for c in cells]
        cells_hot_flat = [[obj for row in c for obj in row if obj] for c in cells_hot]
        cells_cold_flat = [[obj for row in c for obj in row if obj] for c in cells_cold]
        for i in range(3):
            Group(*cells_flat[i], *cells_hot_flat[i], *cells_cold_flat[i]).move_to(centers[i])

        self.play(
            *[FadeIn(obj) for obj in cells_flat[0]],
            run_time=0.4
        )
        self.pause("Make 0th grid appear")

        cells_to_move = [[obj.copy() for obj in row] for row in cells[0]]
        cells_to_move_flat = []
        to_remove = []
        animations = []
        for iy in range(n):
            for ix in range(n):
                cell_to_move = cells_to_move[iy][ix]
                cell_big = cells[1][iy // 2][ix // 2].copy()

                cells_to_move_flat.append(cell_to_move)
                to_remove.append(cell_big)
                animations.append(
                    Transform(cell_to_move, cell_big, replace_mobject_with_target_in_scene=True)
                )
        self.play(
            Group(*cells_to_move_flat).animate.move_to(centers[1]),
            run_time=0.6
        )
        self.play(
            *animations,
            run_time=0.4
        )
        self.remove(*to_remove)
        self.add(*cells_flat[1])
        self.pause("Make 1st grid appear")

        self.play(
            *[FadeIn(obj) for obj in cells_hot_flat[0]],
            *[FadeIn(obj) for obj in cells_cold_flat[0]],
            run_time=0.4
        )

        start = np.array([-px, py, 0.0])
        end = np.array([0.0, -py, 0.0])
        start, end = start + (end - start) * 0.3, end + (start - end) * 0.3
        restrict_arrow = Arrow(stroke_width=8).put_start_and_end_on(start, end).set_color(C_PURPLE).set_cap_style(CapStyleType.ROUND)
        restrict_arrow.set_z_index(11)

        angle = np.arctan2(*(restrict_arrow.get_end() - restrict_arrow.get_start())[:2][::-1])
        restrict_text = Text("Restrict", color=C_PURPLE).scale(0.5).rotate(angle)
        restrict_text.move_to(restrict_arrow).shift(LEFT * 0.36).shift((restrict_arrow.get_end() - restrict_arrow.get_start()) * -0.02)
        restrict_text.set_z_index(11)
        self.hold(0.2)
        self.play(
            self.create_arrow(restrict_arrow),
            FadeIn(restrict_text),
            run_time=0.6
        )
        self.pause("Fill in values for 0th grid, draw restriction arrow")

        cells_hot_to_move = [[obj.copy() for obj in row] for row in cells_hot[0]]
        cells_cold_to_move = [[obj.copy() for obj in row] for row in cells_cold[0]]
        cells_to_move_flat = [obj.copy() for obj in cells_flat[0]]
        cells_all_to_move_flat = cells_to_move_flat[:]
        to_remove = []
        animations = []
        for iy in range(n):
            for ix in range(n):
                cell_hot_to_move = cells_hot_to_move[iy][ix]
                cell_hot_big = cells_hot[1][iy // 2][ix // 2].copy()
                cells_all_to_move_flat.append(cell_hot_to_move)
                to_remove.append(cell_hot_big)
                animations.append(
                    Transform(cell_hot_to_move, cell_hot_big, replace_mobject_with_target_in_scene=True)
                )

                cell_cold_to_move = cells_cold_to_move[iy][ix]
                cell_cold_big = cells_cold[1][iy // 2][ix // 2].copy()
                cells_all_to_move_flat.append(cell_cold_to_move)
                to_remove.append(cell_cold_big)
                animations.append(
                    Transform(cell_cold_to_move, cell_cold_big, replace_mobject_with_target_in_scene=True)
                )

                if ix % 2 != (ix // 2) % 2 or iy % 2 != (iy // 2) % 2:
                    cell_hot_big.set_opacity(0.0)
                    cell_cold_big.set_opacity(0.0)
        for obj in cells_all_to_move_flat:
            obj.set_z_index(obj.get_z_index() + 30)
        self.play(
            Group(*cells_all_to_move_flat).animate.move_to(centers[1]),
            run_time=0.6
        )
        self.pause("Move values from 0th to 1st")

        self.play(
            *[FadeOut(obj) for obj in cells_to_move_flat],
            *animations,
            run_time=0.4
        )
        self.remove(*to_remove)
        self.add(*cells_hot_flat[1], *cells_cold_flat[1])
        self.pause("Restrict from 0th to 1st")

        prolong_arrow = restrict_arrow.copy().put_start_and_end_on(
            np.array([-end[0], end[1], 0.0]),
            np.array([-start[0], start[1], 0.0]),
        )
        prolong_text = Text("Prolong", color=C_PURPLE).scale(0.5).rotate(-angle)
        prolong_text.set_z_index(11)
        prolong_text.move_to(prolong_arrow).shift(RIGHT * 0.36).shift((prolong_arrow.get_end() - prolong_arrow.get_start()) * -0.12)
        self.play(
            *[Transform(a.copy(), b, replace_mobject_with_target_in_scene=True) for a, b in zip(cells_flat[0], cells_flat[2])],
            self.create_arrow(prolong_arrow),
            FadeIn(prolong_text),
            run_time=0.6
        )
        self.pause("Make 2nd grid appear")

        to_move_group = Group(*[obj.copy() for obj in cells_flat[1] + cells_hot_flat[1] + cells_cold_flat[1]])
        for obj in to_move_group:
            obj.set_z_index(obj.get_z_index() + 30)
        to_move_group.set_z_index(5)
        self.play(
            to_move_group.animate.move_to(centers[2]),
            run_time=0.8
        )
        self.pause("Move values from 1st to 2nd")

        self.add(*cells_hot_flat[2], *cells_cold_flat[2])
        self.play(
            FadeOut(to_move_group),
            run_time=0.6
        )
        self.pause("Prolongate from 1st to 2nd")

        spacing_12 = 3.6

        group_ghost = Group(*cells_flat[0])
        group_ghost.generate_target()
        group_ghost.target.scale(1.8).move_to(ORIGIN)
        to_remove = [
            *cells_flat[2],
            *cells_flat[1],
            *cells_hot_flat[2],
            *cells_hot_flat[1],
            *cells_cold_flat[2],
            *cells_cold_flat[1],
            prolong_arrow,
            prolong_text,
            restrict_arrow,
            restrict_text
        ]
        split_line_1 = Line(np.array([8.0 - 0.5, -10.0, 0.0]), np.array([8.0 + 0.5, 10.0, 0.0])).set_stroke(C_LIGHT_GRAY, width=8)
        split_line_2 = split_line_1.copy()
        self.add(split_line_1)
        move_animations = []
        for ct, c, ch, cc in zip(group_ghost.target, cells_flat[0], cells_hot_flat[0], cells_cold_flat[0]):
            for co in [c, ch, cc]:
                co.generate_target()
                co.target.scale_to_fit_height(ct.get_height()).move_to(ct)
                if co == c:
                    co.target.set_fill(C_GRAY)
                else:
                    co.target.set_opacity(0)
                move_animations.append(
                    MoveToTarget(co)
                )

        nr = 40
        dist = 0.45
        ps = self.pseudo_uniform_points(
            nr,
            dist=dist,
            iterations=200,
            seed=1
        )

        point_group = Group()
        for x, y in ps:
            point = Circle(0.1).set_fill(C_GRAY, opacity=1).set_stroke(opacity=0)
            point.move_to((x, y, 0))
            point.set_z_index(5)
            point_group.add(point)
        point_group.scale_to_fit_height(group_ghost.target.get_height())

        edge_group = Group()
        for i in range(nr):
            for j in range(i + 1, nr):
                if np.linalg.norm(ps[i] - ps[j]) < dist * 1.05:
                    edge = Line(point_group[i].get_center(), point_group[j].get_center()).set_stroke(C_LIGHT_GRAY, width=8)
                    edge.set_z_index(4)
                    edge_group.add(edge)

        self.play(
            *move_animations,
            *[FadeOut(obj, scale=0.5) for obj in to_remove],
            run_time=0.6
        )
        self.pause("Focus on initial grid domain")

        group_1 = Group(*cells_flat[0])
        group_2 = Group(*point_group, *edge_group)
        group_2.move_to(RIGHT * (spacing_12 + 8.0))
        question_mark_1_tex = MathTex("\\textbf{?}", color=C_RED).scale(1.6)
        question_mark_1_tex.move_to(np.array([spacing_12, 2.6, 0.0]))
        self.play(
            split_line_1.animate.move_to(ORIGIN),
            group_1.animate.move_to(LEFT * spacing_12),
            group_2.animate.move_to(RIGHT * spacing_12),
            run_time=0.8
        )
        self.remove(*to_remove)
        self.play(
            FadeIn(question_mark_1_tex, scale=2),
            run_time=0.4
        )
        self.pause("Split screen in half")

        spot_tri_mg = [None] * 3
        for i in range(3):
            spot_tri_mg[i] = self.load_image(f"spot_tri_mg_{i}")
        spot_tri_mg[0].scale(0.8)
        spot_tri_mg[0].move_to(RIGHT * (8.0 + SIZE[0] / 6))
        question_mark_2_tex = MathTex("\\textbf{?!}", color=C_RED).scale(1.6)
        question_mark_2_tex.move_to(np.array([SIZE[0] / 3, 3.3, 0.0]))
        self.add(split_line_2, spot_tri_mg[0])
        self.play(
            group_1.animate.scale(0.9).move_to((SIZE[0] / 3) * LEFT),
            group_2.animate.scale(0.9).move_to(ORIGIN),
            question_mark_1_tex.animate.set_x(0.0),
            split_line_1.animate.move_to((SIZE[0] / 6) * LEFT),
            split_line_2.animate.move_to((SIZE[0] / 6) * RIGHT),
            spot_tri_mg[0].animate.move_to((SIZE[0] / 3) * RIGHT),
            run_time=0.8
        )
        self.play(
            FadeIn(question_mark_2_tex, scale=2),
            run_time=0.4
        )
        self.pause("Split screen in thirds")

        colors = [C_BLUE, C_GREEN, C_RED]
        alpha = 0.35
        spot_tri_mg_tinted = [None] * 3
        for i in range(3):
            pixels = spot_tri_mg[i].get_pixel_array().astype(np.float64)
            pixels[:,:,:3] *= (1 - alpha) * np.ones((3,)) + alpha * hex_to_rgb(colors[i]) / 255.0
            spot_tri_mg_tinted[i] = ImageMobject(pixels.astype(np.uint8))
            spot_tri_mg_tinted[i].scale_to_fit_height(spot_tri_mg[0].get_height())
            spot_tri_mg_tinted[i].set_opacity(0.0)

        spot_scale = 1.2
        spot_shift = spot_tri_mg[0].get_x()

        self.play(
            FadeOut(group_1, shift=spot_shift * LEFT),
            FadeOut(split_line_1, shift=spot_shift * LEFT),
            FadeOut(group_2, shift=spot_shift * LEFT),
            FadeOut(question_mark_1_tex, shift=spot_shift * LEFT),
            FadeOut(split_line_2, shift=spot_shift * LEFT),
            FadeOut(question_mark_2_tex, shift=spot_shift * LEFT),
            spot_tri_mg[0].animate.scale(spot_scale).move_to(ORIGIN),
            run_time=0.8
        )
        self.pause("Focus on fine spot")

        spacing_hori = 4.4
        spacing_vert = 1.2

        spot_tri_mg_tinted[0].scale_to_fit_height(spot_tri_mg[0].get_height()).move_to(spot_tri_mg[0])
        self.play(
            spot_tri_mg[0].animate.scale(1 / spot_scale).move_to(np.array([-spacing_hori, spacing_vert, 0.0])),
            spot_tri_mg_tinted[0].animate.scale(1 / spot_scale).move_to(np.array([-spacing_hori, spacing_vert, 0.0])).set_opacity(1.0),
            run_time=0.6
        )
        self.remove(spot_tri_mg[0])

        for i in range(1, 3):
            spot_tri_mg_tinted[i].move_to(spot_tri_mg_tinted[i - 1])

            self.hold(0.2)
            self.play(
                spot_tri_mg_tinted[i].animate.set_opacity(1.0).shift(np.array([spacing_hori, -spacing_vert, 0.0])),
                run_time=0.6
            )
        self.pause("Show coarse spots")

        self.clear(run_time=0.6)

    def animate_slide_gravo_demonstration(self):
        self.pause("Start gravo_demonstration")

        title_tex = self.create_title("Gravo MG")
        title_tex.generate_target()
        title_tex.scale(2.0).move_to(ORIGIN)
        self.play(
            FadeIn(title_tex, shift=UP),
            run_time=0.6,
            rate_func=rush_from
        )
        self.pause("Show \"Gravo MG\" title")

        self.play(
            MoveToTarget(title_tex),
            run_time=0.6
        )
        self.fix(title_tex)

        V = []
        w = 16
        h = 11
        np.random.seed(0)
        for iy in range(h):
            for ix in range(w):
                y = 1.00 * iy + 0.20 * ix + np.random.normal(0, 0.1)
                x = 1.20 * ix - 0.75 * iy + np.random.normal(0, 0.1)
                V.append([x, y, 0])
        V = np.array(V)
        V -= np.mean(V, axis=0)

        E = []
        for iy in range(h):
            for ix in range(w):
                if iy < h - 1:
                    E.append([(iy + 0) * w + (ix + 0), (iy + 1) * w + (ix + 0)])
                if ix < w - 1:
                    E.append([(iy + 0) * w + (ix + 0), (iy + 0) * w + (ix + 1)])
                if iy < h - 1 and ix < w - 1:
                    E.append([(iy + 0) * w + (ix + 0), (iy + 1) * w + (ix + 1)])
        np.random.seed(3)
        np.random.shuffle(E)
        E = np.array(E)

        neigh = [set() for _ in range(V.shape[0])]
        for a, b in E:
            a, b = int(a), int(b)
            neigh[a].add(b)
            neigh[b].add(a)
        T = []
        for a in range(len(neigh)):
            for b in neigh[a]:
                if a >= b:
                    continue
                for c in neigh[a]:
                    if b < c and c in neigh[b]:
                        T.append((a, b, c))
        T = np.array(T)

        light_blue = lerp(C_WHITE, C_BLUE, 0.5)
        light_red = lerp(C_WHITE, C_RED, 0.5)

        vertices_fine = []
        for i in range(V.shape[0]):
            vertex = Circle(0.15, color=C_BLUE).set_stroke(C_WHITE, opacity=1, width=4).set_fill(opacity=1)
            vertex.move_to(V[i])
            vertex.set_z_index(10)
            vertices_fine.append(vertex)
        edges_fine = []
        for i in range(E.shape[0]):
            edge = Line(V[E[i][0]], V[E[i][1]]).set_stroke(light_blue, width=8).set_cap_style(CapStyleType.BUTT)
            edge.set_z_index(6)
            edge.start_index = E[i][0]
            edge.end_index = E[i][1]
            edges_fine.append(edge)
        triangles_fine = []
        centers = []
        for t in T:
            triangle = Polygon(*[V[idx] for idx in t]).set_stroke(light_blue, opacity=1, width=8).set_fill(C_BLUE, opacity=0.35)
            triangle.round_corners(0.02)
            triangle.set_z_index(8)
            triangles_fine.append(triangle)
            centers.append(sum(V[t[j]] for j in range(3)) / 3)

        self.play(
            *[FadeIn(obj) for obj in vertices_fine]
        )
        self.pause("Draw fine vertices")

        self.play(
            *[FadeIn(obj, scale=0.5, target_position=center) for obj, center in zip(triangles_fine, centers)],
            run_time=0.8
        )
        self.hold(0.4)
        self.play(
            *[FadeIn(obj, scale=0) for obj in edges_fine],
            run_time=0.4
        )
        self.pause("Draw fine triangles + edges")

        for edge in edges_fine:
            edge.set_stroke(C_LIGHT_GRAY)
        self.play(
            *[obj.animate.set_stroke(light_red).set_fill(C_RED).scale(0.6, about_point=center) for obj, center in zip(triangles_fine, centers)],
            run_time=0.6
        )
        self.hold(1.2)
        self.play(
            *[FadeOut(obj, scale=0.5, target_position=center) for obj, center in zip(triangles_fine, centers)],
            run_time=0.6
        )
        self.pause("Remove triangles")

        lose = 64
        edges_fine, edges_lose = edges_fine[:-lose], edges_fine[-lose:]
        self.play(
            *[obj.animate.set_stroke(light_red, width=1.6 * obj.get_stroke_width()).scale(0.6) for obj in edges_lose],
            run_time=0.4,
        )
        self.hold(0.8)
        self.play(
            *[obj.animate.set_stroke(width=2.4 * obj.get_stroke_width()).scale(0) for obj in edges_lose],
            run_time=0.8
        )
        self.remove(*edges_lose)
        self.pause("Remove some edges")

        code_diagram = self.create_code_diagram(
            "Sampling",
            "Clustering",
            "Connecting",
            "",
            "Smoothing",
            "",
            "Triangulating",
            "Prolonging",
            z_index=80
        )
        for obj in code_diagram:
            obj[0][0][0].set_stroke(C_ORANGE).set_fill(lerp(C_WHITE, C_ORANGE, 0.5))
        code_diagram.move_to(ORIGIN).to_edge(LEFT)

        self.play(
            FadeIn(code_diagram[0][0], scale=0),
            run_time=0.4
        )
        self.pause("Show step: Sampling")

        Vf = V
        Ef = E[:E.shape[0] - lose]
        neighf = [set() for _ in range(Vf.shape[0])]
        for a, b in Ef:
            a, b = int(a), int(b)
            neighf[a].add(b)
            neighf[b].add(a)

        r = 1.85
        samples = []
        vertices_coarse = []
        removed = set()
        radii = []
        sample_animations = []
        radius_animations = []
        fade_animations = []
        unradius_animations = []

        margin = r + 0.15

        to_sample = [*range(len(vertices_fine))]
        np.random.seed(6)
        np.random.shuffle(to_sample)
        for idx in to_sample:
            if idx in removed:
                continue
            x, y, _ = Vf[idx]
            samples.append(idx)

            visible = -SIZE[0] / 2 - margin < x < SIZE[0] / 2 + margin and -SIZE[1] / 2 - margin < y < SIZE[1] / 2 + margin

            vertex_fine = vertices_fine[idx]
            vertex_coarse = vertex_fine.copy()
            vertex_coarse.set_z_index(25)
            self.add(vertex_coarse)
            if visible:
                sample_animations.append([
                    (lambda obj:
                        lambda: obj.animate.scale(1.5).set_fill(C_RED)
                    )(vertex_coarse)
                ])
            vertices_coarse.append(vertex_coarse)
            vertex_fine.set_fill(C_LIGHT_GRAY).set_stroke(opacity=0).scale(0.8)

            radius = Circle(r).set_stroke(C_RED, width=5, opacity=0.8).set_fill(opacity=0)
            radius.scale(1 / 1000)
            radius.set_z_index(20)
            radius.move_to(vertex_coarse)
            radii.append(radius)

            if visible:
                radius_animations.append([
                    (lambda obj:
                        lambda: obj.animate.scale(1000)
                    )(radius)
                ])

            fade_animations_cur = []
            for idx_n in range(len(vertices_fine)):
                if idx_n == idx:
                    continue
                if np.linalg.norm(Vf[idx] - Vf[idx_n]) > r:
                    continue
                if idx_n in removed:
                    continue

                removed.add(idx_n)
                fade_animations_cur.append(
                    (lambda obj:
                        lambda: obj.animate.set_fill(C_LIGHT_GRAY).set_stroke(opacity=0).scale(0.8)
                    )(vertices_fine[idx_n])
                )
            if visible:
                fade_animations.append(fade_animations_cur)

            if visible:
                unradius_animations.append([
                    (lambda obj:
                        lambda: FadeOut(obj)
                    )(radius),
                    (lambda obj:
                        lambda: obj.animate.set_fill(C_DARK_GRAY)
                    )(vertex_coarse)
                ])

        self.play(
            *[anim() for anim in sample_animations[0]],
            run_time=0.4
        )
        self.pause("Sample first vertex")

        self.play(
            *[anim() for anim in radius_animations[0]],
            run_time=0.8
        )
        self.pause("Extend first radius")

        self.play(
            *[anim() for anim in fade_animations[0]],
            run_time=0.4
        )
        self.pause("Eliminate first neighborhood")

        self.play(
            *[anim() for anim in unradius_animations[0]],
            run_time=0.4
        )
        self.play(
            *[anim() for anim in sample_animations[1]],
            run_time=0.4
        )
        self.pause("Sample second point")

        self.play(
            *[anim() for anim in radius_animations[1]],
            run_time=0.6
        )
        self.play(
            *[anim() for anim in fade_animations[1]],
            run_time=0.3
        )
        self.play(
            *[anim() for anim in unradius_animations[1]],
            run_time=0.3
        )
        self.pause("Eliminate second neighborhood")

        all_animations = [sample_animations[2:], radius_animations[2:], fade_animations[2:], unradius_animations[2:]]
        i = 0
        while True:
            animations = []
            for j in range(4):
                for k in range(len(all_animations[0])):
                    if j + k == i:
                        animations.extend(all_animations[j][k])
            if not animations:
                break
            self.play(
                *[anim() for anim in animations],
                run_time=0.12
            )

            i += 1
        self.pause("Do all samples")

        self.play(
            *[Create(obj) for obj in code_diagram[1][1]],
            code_diagram[0][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            run_time=0.2
        )
        self.play(
            FadeIn(code_diagram[1][0], scale=0),
            run_time=0.4
        )
        self.pause("Show step: Clustering")

        q = [(0.0, 0, idx, -1, j) for j, idx in enumerate(samples)]
        pred = [None] * Vf.shape[0]
        in_cluster = [None] * Vf.shape[0]
        depth = [None] * Vf.shape[0]
        Ec = set()
        while q:
            c, d, cur, prv, idx = heapq.heappop(q)

            if in_cluster[cur] is not None:
                x, y = in_cluster[cur], in_cluster[prv]
                if x != y:
                    Ec.add((x, y))
                continue
            in_cluster[cur] = idx
            pred[cur] = prv
            depth[cur] = d

            for nex in neighf[cur]:
                w = np.linalg.norm(Vf[cur] - Vf[nex])
                heapq.heappush(q, (c + w, d + 1, nex, cur, idx))

        Ec = np.array([[a, b] for a, b in Ec])
        neighc = [set() for _ in range(len(samples))]
        for a, b in Ec:
            a, b = int(a), int(b)
            neighc[a].add(b)
            neighc[b].add(a)

        Vc = Vf[samples, :]

        color_map = [C_RED, C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]

        indices = sorted(range(len(samples)), key=lambda x: Vf[samples[x]][0])
        color_indices = [-1] * len(samples)
        np.random.seed(1)
        for cur in indices:
            available = set(range(len(color_map)))
            for nxt in neighc[cur]:
                available -= {color_indices[nxt]}
            color_indices[cur] = np.random.choice([*available])

        cluster_colors = [color_map[c] for c in color_indices]

        self.play(
            *[obj.animate.set_color(cluster_colors[j]).set_stroke(C_WHITE, width=4, opacity=1) for j, obj in enumerate(vertices_coarse)]
        )
        self.pause("Color points")

        cluster_edges = []
        dijkstra_animations = [([], []) for _ in range(10)]
        for cur in range(Vf.shape[0]):
            prv = pred[cur]
            d = depth[cur]
            clu = in_cluster[cur]

            if d:
                edge = Line(Vf[prv], Vf[cur]).set_stroke(cluster_colors[clu], width=12).set_cap_style(CapStyleType.ROUND)
                edge.set_z_index(15)
                edge.cluster = clu
                cluster_edges.append(edge)
                dijkstra_animations[d][0].append(
                    Create(edge)
                )
                dijkstra_animations[d][1].append(
                    vertices_fine[cur].animate.set_fill(cluster_colors[clu]).set_stroke(C_WHITE, width=4, opacity=1).scale(1.25)
                )
            else:
                vertices_fine[cur].set_fill(cluster_colors[clu]).set_stroke(C_WHITE, width=4, opacity=1).scale(1.25)
        while not dijkstra_animations[-1][0]:
            dijkstra_animations.pop()
        dijkstra_animations = dijkstra_animations[1:]
        for animations_edge, animations_vertex in dijkstra_animations:
            self.play(
                *animations_edge,
                run_time=0.8
            )
            self.play(
                *animations_vertex,
                run_time=0.3
            )
            self.hold(0.2)
        self.pause("Add Dijkstra edges")

        clusters = [[] for _ in range(max(in_cluster) + 1)]
        for idx, c in enumerate(in_cluster):
            clusters[c].append(idx)

        cluster_polygons = []
        for i, cluster in enumerate(clusters):
            color = lerp(C_WHITE, cluster_colors[i], 0.25)

            polygons = Group()
            drawn = set()
            for a in cluster:
                for b in [j for j in cluster if j in neighf[a]]:
                    va, vb = [Vf[j] for j in (a, b)]
                    key = tuple(sorted([a, b]))
                    if len(set(key)) == len(key) and key not in drawn:
                        drawn.add(key)

                        line = Line(va, vb).set_stroke(color, width=64).set_fill(color, opacity=1).set_cap_style(CapStyleType.ROUND)
                        polygons.add(line)

                    for c in [j for j in cluster if j in neighf[a] & neighf[b]]:
                        key = tuple(sorted([a, b, c]))
                        if len(set(key)) == len(key) and key not in drawn:
                            drawn.add(key)

                            vc = Vf[c]
                            eab, ebc = vb - va, vc - vb
                            eab /= np.linalg.norm(eab)
                            ebc /= np.linalg.norm(ebc)
                            if np.linalg.norm(np.cross(eab, ebc)) < 1e-1:
                                continue
                            polygon = Polygon(va, vb, vc).set_stroke(color, width=64).set_fill(color, opacity=1).set_cap_style(CapStyleType.ROUND).round_corners(0.05)
                            polygons.add(polygon)

                    for c in [j for j in cluster if j in neighf[b]]:
                        for d in [j for j in cluster if j in neighf[c]]:
                            if a not in neighf[d] or a in neighf[c] or b in neighf[d]:
                                continue

                            key = tuple(sorted([a, b, c, d]))
                            if len(set(key)) == len(key) and key not in drawn:
                                drawn.add(key)
                                vc, vd = [Vf[j] for j in (c, d)]
                                polygon = Polygon(va, vb, vc, vd).set_stroke(color, width=64).set_fill(color, opacity=1).set_cap_style(CapStyleType.ROUND).round_corners(0.05)
                                polygons.add(polygon)

            for obj in polygons:
                obj.generate_target()
                obj.set_stroke(C_WHITE).set_fill(C_WHITE)
            cluster_polygons.append(polygons)

        cluster_edges_new = []
        connecting_edges = {}
        for idx, (a, b) in enumerate(Ef):
            a, b = int(a), int(b)
            p, q = in_cluster[a], in_cluster[b]
            if p != q:
                key = tuple(sorted([p, q]))
                if key not in connecting_edges:
                    connecting_edges[key] = []
                connecting_edges[key].append(edges_fine[idx])
                continue
            if pred[a] == b or pred[b] == a:
                continue

            edge = Line(Vf[a], Vf[b]).set_stroke(cluster_colors[p], width=12).set_cap_style(CapStyleType.ROUND)
            edge.cluster = p
            edge.set_z_index(20)
            cluster_edges_new.append(edge)
        cluster_edges.extend(cluster_edges_new)
        self.play(
            *[FadeIn(obj, scale=0) for obj in cluster_edges_new],
            *[MoveToTarget(obj) for group in cluster_polygons for obj in group],
            run_time=0.4
        )
        self.pause("Add all cluster edges, show clusters")

        nonconnecting_edges = {*edges_fine}
        for edges in connecting_edges.values():
            nonconnecting_edges -= {*edges}

        self.remove(*nonconnecting_edges)
        self.play(
            *[obj.animate.set_fill(lerp(C_WHITE, str(obj.get_color()), 0.35)) for obj in vertices_fine],
            *[FadeOut(obj) for obj in cluster_edges],
            run_time=0.4
        )
        self.pause("Fade away cluster edges, focus on coarse vertices")

        self.play(
            *[Create(obj) for obj in code_diagram[2][1]],
            code_diagram[1][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            run_time=0.2
        )
        self.play(
            FadeIn(code_diagram[2][0], scale=0),
            run_time=0.4
        )
        self.pause("Show step: Connecting")

        min_key = min(connecting_edges.keys(), key=lambda x: np.linalg.norm(Vc[x[0]] + Vc[x[1]]))
        to_highlight = [cluster_polygons[j] for j in min_key]
        self.play(
            [obj.animate.scale(1.3) for obj in to_highlight],
            run_time=0.4
        )
        self.hold(0.8)
        self.play(
            [obj.animate.scale(1 / 1.3) for obj in to_highlight],
            run_time=0.4
        )
        self.pause("Highlight first two clusters")

        self.play(
            *[obj.animate.set_stroke(C_GRAY, width=12) for obj in connecting_edges[min_key]],
            run_time=0.4
        )
        self.pause("Highlight first connecting edges")

        move_animations = {}
        appear_animations = {}
        edges_coarse = []
        for key, edges in connecting_edges.items():
            a, b = key

            move_animations[key] = [
                obj.animate.put_start_and_end_on(Vc[in_cluster[obj.start_index]], Vc[in_cluster[obj.end_index]]) for obj in edges
            ]
            edge_coarse = Line(Vc[a], Vc[b]).set_stroke(C_GRAY, width=12).set_cap_style(CapStyleType.ROUND)
            edge_coarse.set_z_index(20)
            edge_coarse.start_index = a
            edge_coarse.end_index = b
            edges_coarse.append(edge_coarse)
            appear_animations[key] = [
                FadeIn(edge_coarse, scale=0)
            ]

        self.play(
            *move_animations.pop(min_key),
            run_time=0.8
        )
        self.play(
            *appear_animations.pop(min_key),
            run_time=0.4
        )
        self.remove(*connecting_edges[min_key])
        self.pause("Draw first coarse edge")

        all_move_animations = []
        all_appear_animations = []
        for key in {*connecting_edges.keys()} - {min_key}:
            all_move_animations.extend(move_animations[key])
            all_appear_animations.extend(appear_animations[key])
        self.play(
            *all_move_animations,
            run_time=1.6
        )
        self.play(
            *all_appear_animations,
            run_time=0.6
        )
        self.remove(*edges_fine)
        self.pause("Draw all coarse edges")

        self.play(
            *[Create(obj) for obj in code_diagram[3][1]],
            code_diagram[2][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            run_time=0.2
        )
        self.play(
            FadeIn(code_diagram[3][0], scale=0),
            run_time=0.4
        )
        self.pause("Show step: Smoothing")

        Vc_new = [np.zeros((3,)) for _ in range(len(samples))]
        Vc_new_num = [0] * len(samples)
        for idx, vertex in enumerate(vertices_fine):
            clu = in_cluster[idx]
            Vc_new[clu] += Vf[idx]
            Vc_new_num[clu] += 1
        Vc_new = np.array([p / n for p, n in zip(Vc_new, Vc_new_num)])

        self.play(
            *[obj.animate.set_fill(lerp(C_WHITE, cluster_colors[c], 0.65)) for obj, c in zip(vertices_fine, in_cluster)],
            run_time=0.4
        )
        self.pause("Bring fine points back into focus")

        molds = []
        for idx, vertex_fine in enumerate(vertices_fine):
            clu = in_cluster[idx]
            vertex_coarse = vertices_coarse[clu]

            mold = vertex_fine.copy().set_fill(opacity=0).set_stroke(vertex_fine.get_fill_color(), width=8)
            mold.scale(1.25 * vertex_coarse.width / vertex_fine.width)
            mold.set_z_index(50)
            molds.append(mold)

            mold.generate_target()
            mold.target.set_stroke(cluster_colors[clu])
            mold.target.move_to(Vc_new[clu])
        self.play(
            *[FadeIn(obj, scale=0) for obj in molds],
            run_time=0.6
        )
        self.pause("Highlight fine points for averaging")

        self.play(
            *[MoveToTarget(obj) for obj in molds],
            run_time=0.8
        )
        self.pause("Average fine cluster points")

        self.play(
            *[obj.animate.move_to(Vc_new[idx]) for idx, obj in enumerate(vertices_coarse)],
            *[obj.animate.put_start_and_end_on(Vc_new[obj.start_index], Vc_new[obj.end_index]) for obj in edges_coarse],
            run_time=1.6
        )
        self.play(
            *[FadeOut(obj) for obj in molds],
            run_time=0.6
        )
        self.pause("Move coarse vertices")

        Vc = Vc_new

        self.play(
            *[obj.animate.set_stroke(C_WHITE).set_fill(C_WHITE) for group in cluster_polygons for obj in group],
            *[obj.animate.set_fill(lerp(C_WHITE, C_BLUE, 0.35)) for obj in vertices_fine],
            *[obj.animate.set_fill(C_GREEN) for obj in vertices_coarse],
            run_time=0.6
        )
        self.remove(*cluster_polygons)
        self.pause("Recolor points, fade clusters")

        self.play(
            *[Create(obj) for obj in code_diagram[4][1]],
            code_diagram[3][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            run_time=0.2
        )
        self.play(
            FadeIn(code_diagram[4][0], scale=0),
            run_time=0.4
        )
        self.pause("Show step: Triangulating")

        Tc = []
        for a in range(len(neighc)):
            for b in neighc[a]:
                if a >= b:
                    continue
                for c in neighc[a]:
                    if b < c and c in neighc[b]:
                        Tc.append((a, b, c))
        focus_point_1 = np.array([2.5, 0.0, 0.0])
        Tc = sorted(Tc, key=lambda x: np.linalg.norm(sum(Vc[x[j]] - focus_point_1 for j in range(3))))
        Tc = np.array(Tc)

        highlight_triangle = Polygon(*[Vc[idx] for idx in Tc[0]]).set_stroke(C_GREEN, opacity=1, width=32).set_fill(opacity=0)
        highlight_triangle.set_z_index(50)
        highlight_triangle.round_corners(0.02)
        self.play(
            Create(highlight_triangle),
            run_time=0.6,
            rate_func=rush_from
        )
        self.pause("Highlight 3-clique")

        light_green = lerp(C_WHITE, C_GREEN, 0.5)
        triangles_coarse = []
        centers = []
        for t in Tc:
            triangle = Polygon(*[Vc[idx] for idx in t]).set_stroke(light_green, opacity=1, width=8).set_fill(C_GREEN, opacity=0.35)
            triangle.round_corners(0.02)
            triangle.set_z_index(8)
            triangles_coarse.append(triangle)
            centers.append(sum(Vc[t[j]] for j in range(3)) / 3)

        self.play(
            Uncreate(highlight_triangle),
            run_time=0.6,
            rate_func=rush_into
        )
        self.play(
            FadeIn(triangles_coarse[0], scale=0.5, target_position=centers[0]),
            run_time=0.6
        )
        self.pause("Add first triangle")

        edges_coarse_new = [edge.copy() for edge in edges_coarse]
        for edge in edges_coarse_new:
            edge.set_stroke(light_green, width=8)
            edge.set_z_index(9)
        self.add(*edges_coarse_new)
        self.play(
            *[obj.animate.set_stroke(light_green, width=8, opacity=0) for obj in edges_coarse],
            AnimationGroup(
                *[FadeIn(obj, scale=0.5, target_position=center) for obj, center in zip(triangles_coarse[1:], centers[1:])],
                lag_ratio=0.01,
                run_time=0.6
            ),
            run_time=0.8
        )
        self.remove(*edges_coarse)
        for triangle in triangles_coarse:
            triangle.set_stroke(opacity=0)
        edges_coarse = edges_coarse_new
        self.pause("Draw coarse triangles + edges")

        self.play(
            *[obj.animate.set_fill(C_BLUE) for obj in vertices_fine],
            run_time=0.4
        )
        self.pause("Bring back fine vertices in focus")

        self.play(
            *[Create(obj) for obj in code_diagram[5][1]],
            code_diagram[4][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            run_time=0.2
        )
        self.play(
            FadeIn(code_diagram[5][0], scale=0),
            run_time=0.4
        )
        self.pause("Show step: Prolonging")

        self.fix(code_diagram)

        camera_zoom = 1.7
        camera_shift = np.array([0.7, 0.1, 0.0])

        indices_1 = sorted(range(len(vertices_fine)), key=lambda x: np.linalg.norm(Vf[x] - focus_point_1))
        min_idx_1 = indices_1[0]
        self.play(
            self.camera.frame.animate.scale(1 / camera_zoom).shift(camera_shift),
            *[vertices_fine[indices_1[j]].animate.set_fill(lerp(C_WHITE, C_BLUE, 0.35)) for j in range(1, len(indices_1))],
            run_time=0.8
        )
        self.play(
            Flash(vertices_fine[indices_1[0]], color=C_BLUE),
            run_time=0.8
        )
        self.pause("Focus on one point")

        self.play(
            triangles_coarse[0].animate.set_fill(opacity=0.65),
            run_time=0.4
        )
        self.pause("Highlight prolongation triangle")

        area_colors = [C_RED, C_ORANGE, C_PURPLE]
        light_colors = [lerp(C_WHITE, color, 0.35) for color in area_colors]
        self.play(
            *[vertices_coarse[idx].animate.set_fill(area_colors[j]) for j, idx in enumerate(Tc[0])],
            run_time=0.4
        )
        self.pause("Highlight prolongation vertices")

        area_triangles = []
        for i in range(3):
            vs = [Vf[min_idx_1], Vc[Tc[0][(i + 1) % 3]], Vc[Tc[0][(i + 2) % 3]]]
            triangle = Polygon(*vs).set_stroke(C_WHITE, opacity=1, width=8).set_fill(light_colors[i], opacity=1)
            triangle.round_corners(0.02)
            triangle.set_z_index(9)
            area_triangles.append(triangle)
        self.play(
            *[FadeIn(obj, scale=0, target_position=Vc[Tc[0][j]]) for j, obj in enumerate(area_triangles)],
            run_time=1.6
        )
        self.pause("Draw barycentric triangles")

        prolongation_elements = [[] for _ in range(len(vertices_fine))]
        for vf_idx in range(len(vertices_fine)):
            p = Vf[vf_idx]
            for tc in Tc:
                for i in range(3):
                    a = Vc[tc[(i + 0) % 3]]
                    b = Vc[tc[(i + 1) % 3]]
                    c = Vc[tc[(i + 2) % 3]]

                    bc = c - b
                    bcp = np.array([-bc[1], bc[0], 0.0])
                    bary = np.dot(bcp, p - b) / np.dot(bcp, a - b)
                    if bary < 0.0:
                        break
                else:
                    prolongation_elements[vf_idx] = [*tc]
                    break
            else:
                best_dist = 10 ** 10
                for ec in Ec:
                    a = Vc[ec[0]]
                    b = Vc[ec[1]]

                    eab = b - a
                    w = np.dot(eab, p - a) / np.dot(eab, eab)
                    if w <= 0.0 or w >= 1.0:
                        continue

                    q = a + eab * w
                    dist = np.linalg.norm(q - p)
                    if dist < best_dist:
                        best_dist = dist
                        prolongation_elements[vf_idx] = [*ec]

        prolongation_lines = []
        prolongation_lines_per_vertex = [[] for _ in range(len(vertices_fine))]
        prolongation_animations_per_vertex = [[] for _ in range(len(vertices_fine))]
        for vf_idx in range(len(vertices_fine)):
            lines = [
                Line(Vf[vf_idx], Vc[idx]).set_sheen_direction(Vc[idx] - Vf[vf_idx]).set_stroke([C_BLUE, C_BLUE] + 2 * [vertices_coarse[idx].get_fill_color()], width=6, opacity=0.5).set_cap_style(CapStyleType.ROUND) \
                    .set_z_index(35)
                for idx in prolongation_elements[vf_idx]
            ]
            prolongation_lines.extend(lines)
            prolongation_lines_per_vertex[vf_idx] = lines
            prolongation_animations_per_vertex[vf_idx] = [
                Create(obj) for obj in lines
            ]
        self.play(
            *prolongation_animations_per_vertex[min_idx_1],
            run_time=0.6
        )
        self.pause("Draw first prolongations")

        focus_point_2 = np.array([-0.5, 0.0, 0.0])
        indices_2 = sorted(range(len(vertices_fine)), key=lambda x: np.linalg.norm(Vf[x] - focus_point_2))

        min_idx_2 = indices_2[0]
        min_edge = min(edges_coarse, key=lambda x: np.linalg.norm(Vc[x.start_index] + Vc[x.end_index] - 2 * focus_point_2))
        vf = Vf[min_idx_2]
        vcs = [
            Vc[min_edge.start_index],
            Vc[min_edge.end_index]
        ]
        vertex_fine = vertices_fine[min_idx_2]
        vertices_edges = [
            vertices_coarse[min_edge.start_index],
            vertices_coarse[min_edge.end_index]
        ]

        self.play(
            vertex_fine.animate.set_fill(C_BLUE),
            Flash(vertex_fine, color=C_BLUE),
            run_time=0.8
        )
        self.pause("Highlight second fine point")

        ec01 = vcs[1] - vcs[0]
        p = vcs[0] + ec01 * (np.dot(ec01, vf - vcs[0]) / np.dot(ec01, ec01))
        vertex_fine_ghost = vertex_fine.copy()
        self.play(
            vertex_fine_ghost.animate.set_fill(lerp(C_WHITE, C_BLUE, 0.65)).move_to(p),
            run_time=0.6
        )
        self.pause("Move fine ghost point")

        length_triangles = []
        for i in range(2):
            vs = [vf, p, vcs[1 - i]]
            triangle = Polygon(*vs).set_stroke(C_WHITE, opacity=1, width=8).set_fill(light_colors[i], opacity=1)
            triangle.round_corners(0.02)
            triangle.set_z_index(9)
            length_triangles.append(triangle)
        self.play(
            vertices_edges[0].animate.set_fill(C_RED),
            vertices_edges[1].animate.set_fill(C_ORANGE),
            *[FadeIn(obj, scale=0, target_position=vcs[j]) for j, obj in enumerate(length_triangles)],
            run_time=1.2
        )
        self.hold(0.2)
        prolongation_lines_per_vertex[min_idx_2][1].set_stroke([C_BLUE, C_BLUE, C_ORANGE, C_ORANGE])
        self.play(
            *prolongation_animations_per_vertex[min_idx_2],
            run_time=0.6
        )
        self.pause("Draw length triangles and second prolongations")

        triangles_coarse[0].set_fill(opacity=0.35)
        self.play(
            FadeOut(vertex_fine_ghost),
            *[FadeOut(obj) for obj in area_triangles],
            *[FadeOut(obj) for obj in length_triangles],
            *[obj.animate.set_fill(C_BLUE) for obj in vertices_fine],
            *[obj.animate.set_fill(C_GREEN) for obj in vertices_coarse],
            *[obj.animate.set_stroke([C_BLUE, C_BLUE, C_GREEN, C_GREEN]) for obj in prolongation_lines_per_vertex[min_idx_1] + prolongation_lines_per_vertex[min_idx_2]],
            self.camera.frame.animate.shift(-camera_shift).scale(camera_zoom),
            run_time=1.2
        )
        self.pause("Put camera back")

        animations = []
        for idx, anims in enumerate(prolongation_animations_per_vertex):
            if idx != min_idx_1 and idx != min_idx_2:
                animations.extend(anims)
        for obj in prolongation_lines:
            obj.set_stroke([C_BLUE, C_BLUE, C_GREEN, C_GREEN])
        self.play(
            *animations,
            run_time=1.2
        )
        self.pause("Draw remaining prolongations")

        self.clear(run_time=0.6)

    def animate_slide_tetrahedral_meshes(self):
        self.pause("Start tetrahedral_meshes")

        background_tex = Text("<\\background>", color=BLACK, font="Consolas").scale(1.6)
        self.play(
            FadeIn(background_tex, shift=UP),
            run_time=0.6
        )
        self.pause("Show end of background")

        title_tex = self.create_title("Our Work?")
        title_tex.generate_target()
        title_tex.scale(2.0).move_to(ORIGIN)
        self.play(
            FadeOut(background_tex, shift=UP),
            FadeIn(title_tex, shift=UP),
            run_time=0.6
        )
        self.pause("Replace with title")

        spot_slice_tri = self.load_image("spot_slice_tri").scale(0.8)
        spot_slice_tet = self.load_image("spot_slice_tet").scale(0.8)
        spot_slice_tri.move_to(LEFT * 4.8 + DOWN * 0.2)
        spot_slice_tet.move_to(RIGHT * 4.8 + DOWN * 0.2)
        tri_and_tet = self.load_image("tri_and_tet").scale_to_fit_width(5.0)
        tri_and_tet.move_to(RIGHT * 0.1 + DOWN * 0.2)

        triangular_text = Text("Triangular", color=C_RED).scale(0.72)
        triangular_text.next_to(tri_and_tet, DOWN).shift(UP * 0.3).align_to(tri_and_tet, LEFT)
        tetrahedral_text = Text("Tetrahedral", color=C_GREEN).scale(0.72)
        tetrahedral_text.next_to(tri_and_tet, DOWN).shift(UP * 0.3).align_to(tri_and_tet, RIGHT)

        surfaces_text = Text("Surfaces", color=C_RED).scale(0.72)
        surfaces_text.next_to(tri_and_tet, UP).set_x(triangular_text.get_x())
        volumes_text = Text("Volumes", color=C_GREEN).scale(0.72)
        volumes_text.next_to(tri_and_tet, UP).set_x(tetrahedral_text.get_x())

        self.play(
            MoveToTarget(title_tex),
            run_time=0.6
        )
        self.play(
            FadeIn(spot_slice_tri, shift=UP),
            FadeIn(surfaces_text, shift=UP),
            run_time=0.6
        )
        self.pause("Show spot tri")

        self.play(
            FadeIn(spot_slice_tet, shift=UP),
            FadeIn(volumes_text, shift=UP),
            run_time=0.6
        )
        self.pause("Show spot tet")

        self.play(
            FadeIn(tri_and_tet, shift=UP),
            FadeIn(triangular_text, shift=UP),
            FadeIn(tetrahedral_text, shift=UP),
            run_time=0.6
        )
        self.pause("Show tri and tet")

        width = SIZE[0] / 2 + spot_slice_tet.get_left()[0]
        half_background = Rectangle(C_GRAY, SIZE[1], width).set_fill(C_WHITE, opacity=1).set_stroke(opacity=0)
        half_background.move_to(LEFT * (0.5 * SIZE[0] + 0.5 * width))
        research_questions_text = Text("Research Questions:", color=C_DARK_GRAY).scale(0.6)
        research_questions_text.to_edge(LEFT).set_y(2.4).shift(LEFT * width)
        self.play(
            half_background.animate.shift(RIGHT * width),
            research_questions_text.animate.shift(RIGHT * width),
            run_time=0.8,
            rate_func=rush_from
        )
        self.pause("Hide tri-part")

        research_question_1_text = Text(
            "1. How can Gravo MG be adapted\nfor tetrahedral meshes?",
            line_spacing=0.8,
            color=C_DARK_GRAY,
            t2s={"Gravo MG": ITALIC},
            t2w={"adapted": BOLD, "for tetrahedral meshes": BOLD}
        ).scale(0.6)
        research_question_1_text.next_to(research_questions_text, DOWN).to_edge(LEFT).shift(np.array([1.0, -0.2, 0.0]))
        emoji_1_text = self.load_image("tools_emoji")
        emoji_1_text.move_to(research_question_1_text).to_edge(LEFT)
        self.play(
            FadeIn(research_question_1_text, shift=RIGHT),
            FadeIn(emoji_1_text, shift=RIGHT),
            run_time=0.4,
            rate_func=rush_from
        )
        self.pause("Show RQ 1")

        research_question_2_text = Text(
            "2. What design decisions can be made\nto improve convergence of the\nhierarchies it generates?",
            line_spacing=0.8,
            color=C_DARK_GRAY,
            t2w={"improve convergence": BOLD}
        ).scale(0.6)
        research_question_2_text.next_to(research_question_1_text, DOWN).to_edge(LEFT).shift(np.array([1.0, -0.2, 0.0]))
        emoji_2_text = self.load_image("trend_emoji")
        emoji_2_text.move_to(research_question_2_text).to_edge(LEFT)
        self.play(
            FadeIn(research_question_2_text, shift=RIGHT),
            FadeIn(emoji_2_text, shift=RIGHT),
            run_time=0.4,
            rate_func=rush_from
        )
        self.pause("Show RQ 2")

        research_question_3_text = Text(
            "3. How does the adapted Gravo MG\ncompare to other methods in both\nruntime and convergence?",
            line_spacing=0.8,
            color=C_DARK_GRAY,
            t2s={"Gravo MG": ITALIC},
            t2w={"compare to other methods": BOLD}
        ).scale(0.6)
        research_question_3_text.next_to(research_question_2_text, DOWN).to_edge(LEFT).shift(np.array([1.0, -0.2, 0.0]))
        emoji_3_text = self.load_image("timer_emoji")
        emoji_3_text.move_to(research_question_3_text).to_edge(LEFT)
        self.play(
            FadeIn(research_question_3_text, shift=RIGHT),
            FadeIn(emoji_3_text, shift=RIGHT),
            run_time=0.4,
            rate_func=rush_from
        )
        self.pause("Show RQ 3")

        self.clear(run_time=0.6)

    def animate_slide_contribution_basics(self):
        title_tex = self.create_title("The Essentials")
        self.play(
            FadeIn(title_tex),
            run_time=0.6
        )
        self.pause("Show title")

        code_diagram_z_index = 0
        code_diagram = self.create_code_diagram(
            "Sampling",
            "Clustering",
            "Connecting",
            "",
            "Smoothing",
            "",
            "Triangulating",
            "Prolonging",
            z_index=code_diagram_z_index
        )
        code_diagram.move_to(LEFT * 3.0)
        self.play(
            FadeIn(code_diagram, shift=UP),
            run_time=0.6
        )
        self.pause("Show code diagram")

        self.play(
            *[group[0][0][0].animate.set_fill(lerp(C_WHITE, C_GREEN, 0.5)).set_stroke(C_GREEN) for group in code_diagram[:4]],
            run_time=0.4
        )
        self.pause("Show trivial stages")

        self.play(
            code_diagram[4][0][0][0].animate.set_fill(lerp(C_WHITE, C_BLUE, 0.5)).set_stroke(C_BLUE),
            run_time=0.6
        )
        self.pause("Highlight \"Triangulating\"")

        tetrahedralizing_text = Text("Tetrahedralizing", color=C_BLACK).scale(0.55)
        tetrahedralizing_text.move_to(code_diagram[4][0][0])
        tetrahedralizing_text.set_z_index(code_diagram_z_index + 0.8)
        self.play(
            Transform(code_diagram[4][0][1], tetrahedralizing_text)
        )
        self.pause("Rename to \"Tetrahedralizing\"")

        V = np.array([
            [1.5, 0.0, 0.0],
            [3.5, -0.6, 0.0],
            [4.5, 0.4, 0.0],
            [3.0, 2.4, 0.0],
        ])
        vertices = []
        for i in range(4):
            vertex = Circle(0.01, color=C_BLUE).set_stroke(C_WHITE, opacity=1, width=6).set_fill(opacity=1)
            vertex.move_to(V[i])
            vertex.set_z_index(10)
            vertices.append(vertex)
        Group(*vertices).scale_to_fit_height(4.4).move_to(RIGHT * 3.0)
        for obj in vertices:
            obj.scale_to_fit_height(0.6)

        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edge = Line(vertices[i].get_center(), vertices[j].get_center()).set_stroke(C_LIGHT_GRAY, width=12)
                edge.set_z_index(5 + 3 * (j == 3))
                edges.append(edge)
        edge_backdrop = Line(vertices[1].get_center(), vertices[3].get_center()).set_stroke(C_WHITE, width=18)
        edge_backdrop.set_z_index(7)
        self.play(
            Create(edge_backdrop),
            *[Create(obj) for obj in edges],
            *[FadeIn(obj, scale=0.5) for obj in vertices],
            run_time=0.4
        )
        self.pause("Make vertices and edges appear")

        highlight_triangle = Polygon(*[vertices[idx].get_center() for idx in range(3)]).set_stroke(C_BLUE, opacity=1, width=32).set_fill(opacity=0)
        highlight_triangle.set_z_index(6)
        highlight_triangle.round_corners(0.02)
        self.play(
            Create(highlight_triangle),
            run_time=0.4,
            rate_func=rush_from
        )
        self.play(
            highlight_triangle.animate.set_fill(C_BLUE, opacity=0.5),
            run_time=0.4
        )
        self.pause("Highlight 3-clique triangle")

        top_edges = []
        for i in range(3):
            edge = Line(vertices[i].get_center(), vertices[3].get_center()).set_stroke(C_BLUE, opacity=1, width=32).set_fill(opacity=0)
            edge.set_z_index(9)
            top_edges.append(edge)
        highlight_tetrahedron = Polygon(*[vertices[idx].get_center() for idx in range(4)]).set_stroke(opacity=0.0).set_fill(opacity=0)
        self.play(
            highlight_triangle.animate.set_fill(opacity=0),
            *[Create(obj) for obj in top_edges],
            run_time=0.4
        )
        self.play(
            highlight_tetrahedron.animate.set_fill(C_BLUE, opacity=0.5),
            run_time=0.4
        )
        self.pause("Highlight 4-clique tetrahedron")

        for obj in edges:
            obj.set_stroke(C_BLUE)
        self.play(
            [obj.animate.set_stroke(width=edges[0].get_stroke_width()) for obj in top_edges],
            highlight_triangle.animate.set_stroke(width=edges[0].get_stroke_width()),
            FadeOut(highlight_tetrahedron),
            run_time=0.4,
            rate_func=rush_from
        )
        self.remove(*top_edges, highlight_tetrahedron)
        self.play(
            code_diagram[4][0][0][0].animate.set_fill(lerp(C_WHITE, C_GREEN, 0.5)).set_stroke(C_GREEN),
            code_diagram[5][0][0][0].animate.set_fill(lerp(C_WHITE, C_BLUE, 0.5)).set_stroke(C_BLUE),
            run_time=0.6
        )
        self.pause("Highlight \"Prolonging\"")

        vertex_fine = Circle(0.2, color=C_RED).set_stroke(C_WHITE, opacity=1, width=6).set_fill(opacity=1)
        vertex_fine.move_to(
            0.35 * vertices[0].get_center() + \
            0.25 * vertices[1].get_center() + \
            0.15 * vertices[2].get_center() + \
            0.25 * vertices[3].get_center()
        )
        self.play(
            FadeIn(vertex_fine, scale=0.0),
            run_time=0.4
        )
        self.play(
            Flash(vertex_fine, color=C_RED),
            run_time=0.6
        )
        self.pause("Draw fine vertex")

        prolongation_lines = []
        for i in range(4):
            pf, pc = vertex_fine.get_center(), vertices[i].get_center()
            line = Line(pf, pc) \
                .set_sheen_direction(pc - pf) \
                .set_stroke([C_RED, C_RED, C_BLUE, C_BLUE], width=10, opacity=0.5) \
                .set_cap_style(CapStyleType.ROUND)
            line.set_z_index(15)
            prolongation_lines.append(line)
        self.play(
            *[Create(obj) for obj in prolongation_lines],
            run_time=0.8
        )
        self.pause("Draw prolongation lines")

        self.clear(run_time=0.6)

    def animate_slide_experiments(self):
        self.pause()

        bueno = self.load_image("bueno")
        bueno.scale_to_fit_height(1e-3)
        bueno.set_z_index(80)
        self.play(
            bueno.animate.scale_to_fit_height(5.0),
            run_time=0.8,
            rate_func=linear
        )
        self.pause("Bueno!")

        self.clear(run_time=0.6)

    def animate_slide_boundary(self):
        cube_boundary = self.load_image("cube_boundary")
        cube_boundary.scale_to_fit_height(6.0)
        cube_boundary.move_to(DOWN * 0.5)
        cube_boundary.set_z_index(25)
        self.play(
            FadeIn(cube_boundary),
            run_time=0.6
        )
        self.hold(0.4)

        order_texts_data = [
            ("3. Corner", C_ORANGE),
            ("2. Ridge", C_GREEN),
            ("1. Surface", C_BLUE),
            ("0. Interior", C_GRAY)
        ]
        order_texts = []
        for i, (s, c) in enumerate(order_texts_data):
            text = Text(s, color=lerp(C_BLACK, c, 0.75), weight=BOLD)
            text.to_edge(LEFT).set_y(i * -1.0 + (i == 1) * -0.06)
            text.set_z_index(30)
            order_texts.append(text)
        Group(*order_texts).next_to(cube_boundary, RIGHT).shift(2 * LEFT)

        self.play(
            AnimationGroup(
                *[FadeIn(obj, shift=2 * RIGHT) for obj in order_texts],
                lag_ratio=0.2
            ),
            cube_boundary.animate.shift(2 * LEFT),
            run_time=0.6
        )
        self.pause("Show cube")

        self.clear(run_time=0.6)

    def animate_slide_optimization_1_vertex_ordering(self):
        title_tex = self.create_title("Optimization 1: Sampling Order")
        self.play(
            FadeIn(title_tex),
            run_time=0.6
        )
        self.fix(title_tex)
        self.pause("Show title")

        nr = 40
        dist = 0.45
        ps = self.pseudo_uniform_points(
            nr=nr,
            dist=dist,
            iterations=200,
            seed=1
        )

        vertices_fine = []
        for x, y in ps:
            vertex_fine = Circle(0.08).set_fill(C_GRAY, opacity=1).set_stroke(opacity=0)
            vertex_fine.move_to((x, y, 0))
            vertex_fine.set_z_index(5)
            vertices_fine.append(vertex_fine)
        Group(*vertices_fine).scale_to_fit_height(5.0).move_to(0.1 * DOWN)

        edges_fine = []
        for i in range(nr):
            for j in range(i + 1, nr):
                if np.linalg.norm(ps[i] - ps[j]) < dist * 1.05:
                    edge_fine = Line(vertices_fine[i].get_center(), vertices_fine[j].get_center()).set_stroke(C_LIGHT_GRAY, width=12).set_cap_style(CapStyleType.ROUND)
                    edge_fine.set_z_index(4)
                    edges_fine.append(edge_fine)

        self.play(
            *[FadeIn(obj, scale=0.0) for obj in vertices_fine + edges_fine],
            run_time=0.6
        )
        self.pause("Make domain appear")

        r = 1.2

        for it in range(2):
            for obj in vertices_fine + edges_fine:
                obj.save_state()

            samples = []
            vertices_coarse = []
            removed = set()
            radii = []
            sample_animations = []
            radius_animations = []
            fade_animations = []
            unradius_animations = []

            boundary_colors = [C_GRAY, C_BLUE, C_GREEN]
            boundary = [0] * len(vertices_fine)
            if it == 1:
                boundary = [sum(abs(float(c)) > 0.99 for c in ps[j]) for j in range(len(vertices_fine))]

                self.play(
                    *[obj.animate.set_fill(boundary_colors[b]) for obj, b in zip(vertices_fine, boundary) if b],
                    run_time=0.4
                )
                self.play(
                    *[Flash(obj, color=boundary_colors[b]) for obj, b in zip(vertices_fine, boundary) if b],
                    run_time=0.6
                )
                self.pause("Flash boundary")

                order_circles = []
                for i in range(len(boundary_colors)):
                    circle = Circle(0.3).set_stroke(opacity=0).set_fill(boundary_colors[len(boundary_colors) - 1 - i], opacity=1)
                    circle.shift(3.0 * (i - (len(boundary_colors) - 1) / 2) * RIGHT).to_edge(DOWN)
                    order_circles.append(circle)

                order_arrows = []
                for i in range(len(order_circles) - 1):
                    arrow = Arrow(order_circles[i], order_circles[i + 1], color=C_DARK_GRAY)
                    order_arrows.append(arrow)

                animations = []
                for i in range(len(order_circles)):
                    if i:
                        animations.append(
                            self.create_arrow(order_arrows[i - 1]))

                    order_circles[i].scale(0.001)
                    animations.append(
                        order_circles[i].animate.scale(1000)
                    )

                self.play(
                    AnimationGroup(
                        *animations,
                        lag_ratio=0.4
                    ),
                    run_time=0.8
                )
                self.pause("Show order")

            to_sample = [*range(len(vertices_fine))]
            np.random.seed(4 * it)
            np.random.shuffle(to_sample)
            to_sample = sorted(to_sample, key=lambda x: -boundary[x])
            for idx in to_sample:
                if idx in removed:
                    continue
                x, y, _ = vertices_fine[idx].get_center()
                samples.append(idx)

                vertex_fine = vertices_fine[idx]
                vertex_coarse = vertex_fine.copy()
                vertex_coarse.set_z_index(25)
                vertices_coarse.append(vertex_coarse)

                self.add(vertex_coarse)
                sample_animations.append([
                    (lambda obj:
                        lambda: obj.animate.scale(1.5).set_fill(C_RED)
                    )(vertex_coarse)
                ])

                radius = Circle(r).set_stroke(C_RED, width=5, opacity=0.8).set_fill(opacity=0)
                radius.scale(1 / 1000)
                radius.set_z_index(20)
                radius.move_to(vertex_coarse)
                radii.append(radius)

                radius_animations.append([
                    (lambda obj:
                        lambda: obj.animate.scale(1000)
                    )(radius)
                ])

                fade_animations_cur = []
                for idx_n in range(len(vertices_fine)):
                    if np.linalg.norm(vertices_fine[idx].get_center() - vertices_fine[idx_n].get_center()) > r:
                        continue
                    if idx_n in removed:
                        continue

                    removed.add(idx_n)
                    fade_animations_cur.append(
                        (lambda obj:
                            lambda: obj.animate.set_fill(C_LIGHT_GRAY).set_stroke(opacity=0).scale(0.8)
                        )(vertices_fine[idx_n])
                    )
                fade_animations.append(fade_animations_cur)

                unradius_animations.append([
                    (lambda obj:
                        lambda: FadeOut(obj)
                    )(radius),
                    (lambda obj, col:
                        lambda: obj.animate.set_fill(lerp(C_BLACK, col, 0.5))
                    )(vertex_coarse, boundary_colors[boundary[idx]])
                ])

            all_animations = [sample_animations, radius_animations, fade_animations, unradius_animations]
            for b in sorted(set(boundary))[::-1]:
                i = -1
                animated_something = False
                while True:
                    i += 1

                    animations = []
                    for j in range(4):
                        for k in range(len(all_animations[0])):
                            if boundary[samples[k]] != b:
                                continue
                            if j // 3 + k == i:
                                animations.extend(all_animations[j][k])
                    if not animations:
                        if not animated_something:
                            continue
                        else:
                            break
                    animated_something = True

                    self.play(
                        *[anim() for anim in animations],
                        run_time=0.08
                    )

                debug_indices = []
                if not FINAL:
                    for i in range(len(vertices_coarse)):
                        debug_index = Text(f"{i}", color=C_RED)
                        debug_index.move_to(vertices_coarse[i])
                        debug_index.set_z_index(1000)
                        debug_indices.append(debug_index)
                self.add(*debug_indices)
                self.pause(f"Do all samples of boundary {b}")
                self.remove(*debug_indices)

            conn_r = 2.4
            to_connect = set()
            for i in range(len(vertices_coarse)):
                v_i = vertices_coarse[i].get_center()
                for j in range(i + 1, len(vertices_coarse)):
                    v_j = vertices_coarse[j].get_center()

                    if np.linalg.norm(v_j - v_i) < conn_r:
                        to_connect.add((i, j))

            if it == 1:
                to_connect = to_connect \
                    - {(4, 9)} \
                    | {(2, 9), (1, 10)}

            edges_coarse = []
            for i, j in to_connect:
                if np.random.rand() < 0.5:
                    i, j = j, i
                v_j, v_i = vertices_coarse[i].get_center(), vertices_coarse[j].get_center()
                edge_coarse = Line(v_i, v_j).set_stroke(C_LIGHT_GRAY, width=16).set_cap_style(CapStyleType.ROUND)
                edge_coarse.set_z_index(24)
                edges_coarse.append(edge_coarse)

            self.play(
                *[obj.animate.set_fill(C_LIGHT_LIGHT_GRAY) for obj in vertices_fine],
                *[obj.animate.set_stroke(opacity=0) for obj in edges_fine],
                *[Create(obj) for obj in edges_coarse],
                run_time=0.6
            )
            self.pause("Draw coarse edges")

            if it == 1:
                break

            self.play(
                *[FadeOut(obj) for obj in vertices_coarse + edges_coarse],
                *[obj.animate.restore() for obj in vertices_fine + edges_fine],
                run_time=0.8
            )
            self.pause("Restore state")

        mg_positions = [[np.array([4.0 * (j - 1), -0.6 + 1.5 * (1 - 2 * i), 0.0]) for j in range(3)] for i in range(2)]
        mg_images = [[None] * 3 for _ in range(2)]
        mg_arrows = [[None] * 2 for _ in range(2)]
        for i, name in enumerate(["random", "rank"]):
            for j in range(3):
                image = self.load_image(f"cube_mg_order_{name}_{j}")
                image.scale_to_fit_height(2.8)
                image.move_to(mg_positions[i][j])
                image.set_z_index(60)
                mg_images[i][j] = image

                if j:
                    start = mg_positions[i][j - 1]
                    diff = mg_positions[i][j] - start
                    arrow = Arrow(color=C_DARK_GRAY).put_start_and_end_on(start + 0.35 * diff, start + 0.65 * diff)
                    arrow.set_z_index(60)
                    mg_arrows[i][j - 1] = arrow

        white_rectangle = Rectangle(C_WHITE, SIZE[1] + 1, SIZE[0] + 1).set_fill(opacity=0.85)
        white_rectangle.set_z_index(50)
        self.play(
            FadeIn(white_rectangle),
            *[FadeIn(obj, shift=UP * 0.5) for obj in mg_images[0]],
            *[FadeIn(obj, shift=UP * 0.5) for obj in mg_arrows[0]],
            run_time=0.6
        )
        self.pause("Show random order MG")

        self.play(
            *[FadeIn(obj, shift=UP * 0.5) for obj in mg_images[1]],
            *[FadeIn(obj, shift=UP * 0.5) for obj in mg_arrows[1]],
            run_time=0.6
        )
        self.pause("Show rank order MG")

        self.clear(run_time=0.6)

    def animate_slide_optimization_2_sampling_density(self):
        title_tex = self.create_title("Optimization 2: Sampling Density")
        self.play(
            FadeIn(title_tex),
            run_time=0.6
        )
        self.fix(title_tex)
        self.pause("Show title")

        self.clear(run_time=0.6)

    def animate_slide_optimization_3_pit_prevention(self):
        title_tex = self.create_title("Optimization 3: Pit Prevention")
        self.play(
            FadeIn(title_tex),
            run_time=0.6
        )
        self.fix(title_tex)
        self.pause("Show title")

        image_before = self.load_image(f"cube_mg_pit_0").scale_to_fit_height(4.4)
        image_before.shift(np.array([-3.0, -0.2, 0.0]))
        image_after = self.load_image(f"cube_mg_pit_noprev_1").scale_to_fit_height(4.4)
        image_after.shift(np.array([3.0, -0.2, 0.0]))

        start = image_before.get_center()
        diff = image_after.get_center() - start
        arrow = Arrow(color=C_DARK_GRAY).put_start_and_end_on(start + 0.35 * diff, start + 0.65 * diff)

        self.play(
            FadeIn(image_before, shift=UP * 0.5),
            run_time=0.6
        )
        self.pause("Show before image")

        self.play(
            self.create_arrow(arrow),
            run_time=0.4
        )
        self.play(
            FadeIn(image_after, shift=UP * 0.5),
            run_time=0.6
        )
        self.pause("Show after image")

        V = np.array([
            [ 0.0,  0.0, 0.0],
            [ 0.0, -0.8, 0.0],
            [-1.2, -0.2, 0.0],
            [ 1.2, -0.2, 0.0],
            [-2.2, -0.3, 0.0],
            [ 2.2, -0.3, 0.0],
            [-1.4, -1.2, 0.0],
            [ 1.4, -1.2, 0.0],
            [-0.6, -1.8, 0.0],
            [ 0.6, -1.8, 0.0],
        ])
        E = np.array([
            [0, 1], [0, 2], [0, 3],
            [2, 4], [2, 6], [4, 6],
            [1, 2], [1, 3], [1, 6], [1, 7], [1, 8], [1, 9],
            [6, 8], [7, 9], [8, 9],
            [3, 5], [3, 7], [5, 7]
        ])

        vertices_fine = []
        scale = 2.0
        for i in range(V.shape[0]):
            opacity = i < 4
            vertex = Circle(0.24 / scale, color=C_GRAY).set_stroke(C_WHITE, opacity=opacity, width=4).set_fill(opacity=opacity)
            vertex.move_to(V[i])
            vertex.set_z_index(10)
            vertices_fine.append(vertex)
        Group(*vertices_fine).move_to(DOWN * 0.5).scale(scale)

        edges_fine = []
        for i in range(E.shape[0]):
            a, b = E[i]
            opacity = [a < 4] * 2 + [b < 4] * 2
            edge = Line(vertices_fine[a].get_center(), vertices_fine[b].get_center()).set_cap_style(CapStyleType.ROUND) \
                .set_sheen_direction(vertices_fine[b].get_center() - vertices_fine[a].get_center()).set_stroke(C_LIGHT_GRAY, opacity=opacity, width=12)
            edge.set_z_index(6)
            edge.start_index = a
            edge.end_index = b
            edges_fine.append(edge)

        self.play(
            FadeOut(image_before),
            FadeOut(arrow),
            FadeOut(image_after),
            *[FadeIn(obj, scale=0.0) for obj in vertices_fine + edges_fine],
            run_time=0.6
        )

        debug_indices = []
        if not FINAL:
            for i in range(len(vertices_fine)):
                debug_index = Text(f"{i}", color=C_RED)
                debug_index.move_to(vertices_fine[i])
                debug_index.set_z_index(1000)
                debug_indices.append(debug_index)
        self.add(*debug_indices)
        self.pause("Make domain appear")
        self.remove(*debug_indices)

        vertices_boundary = sorted(sorted(vertices_fine, key=lambda obj: -obj.get_y())[:3], key=lambda obj: obj.get_x())
        edges_boundary = sorted(sorted(edges_fine, key=lambda obj: -obj.get_y())[:4], key=lambda obj: obj.get_x())

        animations = []
        for i in range(4):
            animations.append(
                edges_boundary[i].animate.set_stroke(C_BLUE)
            )

            if i < len(vertices_boundary):
                animations.append(
                    vertices_boundary[i].animate.scale(1.35).set_fill(C_BLUE)
                )

        self.play(
            AnimationGroup(
                *animations,
                lag_ratio=0.1
            ),
            run_time=0.6
        )
        self.hold(0.6)
        self.play(
            *[obj.animate.scale(1 / 1.35) for obj in vertices_boundary],
            *[obj.animate.set_stroke(C_LIGHT_GRAY) for obj in edges_boundary],
            run_time=0.4
        )
        self.pause("Highlight boundary")

        self.play(
            *[obj.animate.scale(1.35) for obj in vertices_fine[1:4]],
            run_time=0.4
        )
        self.pause("Sample points")

        cluster_colors = [C_GRAY, C_BLUE, C_BLUE]
        in_cluster = [-1, 0, 1, 2, 1, 2, 1, 2, 0, 0]

        neigh = {idx: set() for idx in range(V.shape[0])}
        for a, b in E:
            neigh[a].add(b)
            neigh[b].add(a)

        clusters = [[] for _ in range(max(in_cluster) + 1)]
        for idx, c in enumerate(in_cluster):
            if c > -1:
                clusters[c].append(idx)

        cluster_polygons = []
        for i, (a, b, c) in enumerate([
            (1, 8, 9),
            (2, 4, 6),
            (3, 7, 5),
        ]):
            color = lerp(C_WHITE, cluster_colors[i], 0.25)
            color = [color] * 2 + [C_WHITE] * (2 + (i > 0))

            va, vb, vc = [vertices_fine[j].get_center() for j in (a, b, c)]
            ebc = vc - vb
            alpha = np.dot(va - vb, ebc) / np.dot(ebc, ebc)
            vd = vb + alpha * ebc

            polygon = Polygon(va, vb, vc).set_cap_style(CapStyleType.ROUND).round_corners(0.05)
            polygon.set_stroke(color, width=128).set_fill(color, opacity=1)
            polygon.set_sheen_direction(vd - va)
            polygon.set_z_index(3)

            polygon.generate_target()
            polygon.set_stroke(C_WHITE).set_fill(C_WHITE)
            cluster_polygons.append(polygon)

        self.play(
            *[MoveToTarget(obj) for group in cluster_polygons for obj in group],
            run_time=0.4
        )
        self.pause("Show clusters")

        to_restore = [
            *vertices_fine,
            *edges_fine,
            *cluster_polygons
        ]
        for obj in to_restore:
            obj.save_state()

        for it in range(2):
            if it:
                gradient = [C_BLUE] * 2 + [C_GRAY] * 2

                arrow_down = Arrow(stroke_width=12).put_start_and_end_on(
                    vertices_fine[0].get_center() + UP * 0.1,
                    vertices_fine[1].get_center() + DOWN * 0.1,
                ).set_stroke(gradient).set_fill(C_GRAY).set_sheen_direction(DOWN)
                arrow_down.shift(RIGHT * 0.5)
                arrow_down.set_z_index(30)

                cross = Group(
                    Group(
                        Line(np.array([0, 1, 0]), np.array([1, 0, 0])).set_stroke(C_WHITE, width=12).set_z_index(34),
                        Line(np.array([0, 1, 0]), np.array([1, 0, 0])).set_stroke(C_RED, width=8).set_z_index(35)
                    ),
                    Group(
                        Line(np.array([1, 1, 0]), np.array([0, 0, 0])).set_stroke(C_WHITE, width=12).set_z_index(34),
                        Line(np.array([1, 1, 0]), np.array([0, 0, 0])).set_stroke(C_RED, width=8).set_z_index(35)
                    )
                ).scale_to_fit_height(0.4)
                cross.move_to(arrow_down).shift(RIGHT * 0.4)

                self.play(
                    self.create_arrow(arrow_down),
                    run_time=0.6
                )
                self.pause("Draw down arrow")

                for group in cross:
                    self.play(
                        *[Create(obj) for obj in group],
                        run_time=0.2
                    )
                self.pause("Disapprove of down arrow")

                checkmark_template = Group(
                    Group(
                        Line(np.array([0, 1 / 2, 0]), np.array([1 / 3, 0, 0])).set_stroke(C_WHITE, width=12).set_z_index(34),
                        Line(np.array([0, 1 / 2, 0]), np.array([1 / 3, 0, 0])).set_stroke(C_GREEN, width=8).set_z_index(35)
                    ),
                    Group(
                        Line(np.array([1 / 3, 0, 0]), np.array([1, 1, 0])).set_stroke(C_WHITE, width=12).set_z_index(34),
                        Line(np.array([1 / 3, 0, 0]), np.array([1, 1, 0])).set_stroke(C_GREEN, width=8).set_z_index(35)
                    )
                ).scale_to_fit_height(0.4)

                arrows_side = []
                checkmarks = []
                for i in range(2):
                    start = vertices_fine[0].get_center()
                    end = vertices_fine[2 + i].get_center()

                    para = end - start
                    para /= np.linalg.norm(para)
                    perp = np.array([-para[1], para[0], 0.0])
                    if i == 0:
                        perp *= -1
                    mid = (start + end) / 2
                    length = arrow_down.get_length()

                    arrow_side = Arrow(max_stroke_width_to_length_ratio=100).put_start_and_end_on(
                        mid - para * (length / 2),
                        mid + para * (length / 2)
                    ).set_stroke(C_BLUE).set_fill(C_BLUE)
                    arrow_side.shift(perp * 0.5)
                    arrow_side.set_z_index(30)
                    arrows_side.append(arrow_side)

                    checkmark = checkmark_template.copy()
                    checkmark.move_to(arrow_side).shift(perp * 0.4)
                    checkmarks.append(checkmark)

                self.play(
                    *[self.create_arrow(obj) for obj in arrows_side],
                    run_time=0.6
                )
                self.hold(0.2)
                for group_0, group_1 in zip(checkmarks[0], checkmarks[1]):
                    self.play(
                        *[Create(obj) for obj in group_0],
                        *[Create(obj) for obj in group_1],
                        run_time=0.2
                    )
                self.pause("Draw side arrows")

                arrow_up = Arrow(stroke_width=12).put_start_and_end_on(
                    vertices_fine[1].get_center() + DOWN * 0.1,
                    vertices_fine[0].get_center() + UP * 0.1
                ).set_stroke(gradient).set_fill(C_BLUE).set_sheen_direction(DOWN)
                arrow_up.shift(LEFT * 0.5)
                arrow_up.set_z_index(30)

                checkmarks.append(checkmark_template.copy())
                checkmarks[2].move_to(arrow_up).shift(LEFT * 0.4)

                self.play(
                    self.create_arrow(arrow_up),
                    run_time=0.6
                )
                self.hold(0.2)
                for group in checkmarks[2]:
                    self.play(
                        *[Create(obj) for obj in group],
                        run_time=0.2
                    )
                self.pause("Approve of the other way")

                self.play(
                    FadeOut(arrow_down),
                    FadeOut(arrow_up),
                    *[FadeOut(obj) for obj in arrows_side],
                    FadeOut(cross),
                    *[FadeOut(obj) for obj in checkmarks],
                    run_time=0.6
                )

            start = vertices_fine[0].get_center()
            diff = vertices_fine[1 + it].get_center() - start
            cluster_extra = Line(start, start + 1.05 * diff).set_cap_style(CapStyleType.ROUND)
            cluster_extra.set_stroke(lerp(C_WHITE, cluster_colors[it], 0.25), width=128)
            cluster_extra.set_z_index(2)
            self.play(
                self.create_arrow(cluster_extra),
                run_time=0.6
            )
            self.pause("Draw extra cluster connection")

            if it:
                self.play(
                    edges_boundary[2].animate.set_stroke(C_GRAY, width=18),
                    run_time=0.4
                )
                self.pause("Highlight connecting edge")

            Vc = np.array([
                vertices_fine[1].get_center(),
                vertices_fine[2].get_center(),
                vertices_fine[3].get_center(),
            ])

            in_cluster[0] = it

            connecting_edges = {}
            for edge in edges_fine:
                key = tuple(sorted(in_cluster[idx] for idx in [edge.start_index, edge.end_index]))
                if key[0] != key[1]:
                    if key not in connecting_edges:
                        connecting_edges[key] = []
                    connecting_edges[key].append(edge)

            move_animations = []
            for key, edges in connecting_edges.items():
                a, b = key
                move_animations.extend([
                    obj.animate.put_start_and_end_on(Vc[in_cluster[obj.start_index]], Vc[in_cluster[obj.end_index]]) for obj in edges
                ])

            nonconnecting_edges = {*edges_fine} - {edges_boundary[0], edges_boundary[-1]}
            for edges in connecting_edges.values():
                nonconnecting_edges -= {*edges}

            self.play(
                *[obj.animate.set_stroke(opacity=0.0) for obj in nonconnecting_edges],
                vertices_fine[0].animate.set_fill(lerp(C_WHITE, C_BLUE, 0.35)),
                *move_animations,
                run_time=1.2
            )

            edges_remaining = {
                edges_boundary[0],
                edges_boundary[-1],
                *[v[0] for v in connecting_edges.values()]
            }

            appear_animations = []
            edges_coarse = []
            for edge in edges_remaining:
                edge_coarse = edge.copy().set_stroke(C_GRAY, width=18).set_cap_style(CapStyleType.ROUND)
                edge_coarse.set_z_index(9)
                appear_animations.extend([
                    FadeIn(edge_coarse, scale=0)
                ])
                edges_coarse.append(edge_coarse)
            self.play(
                *appear_animations,
                run_time=0.6
            )
            if it == 0:
                self.pause("Draw all coarse edges")
            else:
                self.hold(0.2)

            hole_polygon = Polygon(*Vc).set_stroke(opacity=0.0).set_fill([C_RED, C_GREEN][it], opacity=0.5)
            hole_polygon.set_z_index(4)
            self.play(
                FadeIn(hole_polygon, scale=0.0),
                run_time=0.6
            )
            self.pause("Show hole")

            if it == 0:
                self.play(
                    *[FadeOut(obj) for obj in edges_coarse],
                    FadeOut(hole_polygon),
                    FadeOut(cluster_extra),
                    [obj.animate.restore() for obj in to_restore],
                    run_time=0.8
                )
                self.pause("Restore scene")

        mg_positions = [np.array([4.8 * (min(1, j) - 0.5), -0.6 + 1.5 * ((j + 1) % 3 - 1), 0.0]) for j in range(3)]
        mg_images = [None] * 3
        mg_arrows = [None] * 2
        for i, name in enumerate(["0", "noprev_1", "prev_1"]):
            image = self.load_image(f"cube_mg_pit_{name}")
            image.scale_to_fit_height(2.8)
            image.move_to(mg_positions[i])
            image.set_z_index(60)
            mg_images[i] = image

            if i:
                start = mg_positions[0]
                diff = mg_positions[i] - start
                arrow = Arrow(color=C_DARK_GRAY).put_start_and_end_on(start + 0.35 * diff, start + 0.65 * diff)
                arrow.set_z_index(60)
                mg_arrows[i - 1] = arrow

        white_rectangle = Rectangle(C_WHITE, SIZE[1] + 1, SIZE[0] + 1).set_fill(opacity=0.85)
        white_rectangle.set_z_index(50)
        self.play(
            FadeIn(white_rectangle),
            *[FadeIn(obj, shift=UP * 0.5) for obj in [*mg_images[:2], mg_arrows[0]]],
            run_time=0.6
        )
        self.pause("Show no pit prevention MG")

        self.play(
            *[FadeIn(obj, shift=UP * 0.5) for obj in [mg_images[2]]],
            Transform(mg_arrows[0], mg_arrows[1]),
            run_time=0.6
        )
        self.pause("Show pit prevention MG")

        self.clear(run_time=0.6)

    def animate_slide_optimization_4_boundary_aware_smoothing(self):
        title_tex = self.create_title("Optimization 4: Boundary-Aware Smoothing")
        self.play(
            FadeIn(title_tex),
            run_time=0.6
        )
        self.fix(title_tex)
        self.pause("Show title")

        nr = 40
        dist = 0.45
        ps = self.pseudo_uniform_points(
            nr=nr,
            dist=dist,
            iterations=200,
            seed=1
        )

        boundary_colors = [C_GRAY, C_BLUE, C_GREEN]
        boundary = [sum(abs(float(c)) > 0.99 for c in ps[j]) for j in range(ps.shape[0])]

        vertices_fine = []
        for i, (x, y) in enumerate(ps):
            vertex_fine = Circle(0.08).set_fill(boundary_colors[boundary[i]], opacity=1).set_stroke(opacity=0)
            vertex_fine.move_to((x, y, 0))
            vertex_fine.set_z_index(5)
            vertices_fine.append(vertex_fine)
        Group(*vertices_fine).scale_to_fit_height(5.0).move_to(0.2 * DOWN)

        edges_fine = []
        neighf = {idx: set() for idx in range(nr)}
        for i in range(nr):
            for j in range(i + 1, nr):
                if np.linalg.norm(ps[i] - ps[j]) < dist * 1.05:
                    edge_fine = Line(vertices_fine[i].get_center(), vertices_fine[j].get_center()).set_stroke(C_LIGHT_GRAY, width=12).set_cap_style(CapStyleType.ROUND)
                    edge_fine.start_index = i
                    edge_fine.end_index = j
                    edge_fine.set_z_index(4)
                    edges_fine.append(edge_fine)

                    neighf[i].add(j)
                    neighf[j].add(i)

        self.play(
            *[FadeIn(obj, scale=0.0) for obj in vertices_fine + edges_fine],
            run_time=0.6
        )
        self.pause("Make domain appear")

        r = 1.2

        samples = []
        vertices_coarse = []
        removed = set()

        to_sample = [*range(len(vertices_fine))]
        np.random.seed(4)
        np.random.shuffle(to_sample)
        to_sample = sorted(to_sample, key=lambda x: -boundary[x])
        for idx in to_sample:
            if idx in removed:
                continue
            x, y, _ = vertices_fine[idx].get_center()
            samples.append(idx)

            vertex_fine = vertices_fine[idx]
            vertex_coarse = vertex_fine.copy()
            vertex_coarse.set_z_index(25)
            vertices_coarse.append(vertex_coarse)
            self.add(vertex_coarse)

            for idx_n in range(len(vertices_fine)):
                if np.linalg.norm(vertices_fine[idx].get_center() - vertices_fine[idx_n].get_center()) > r:
                    continue
                if idx_n in removed:
                    continue
                removed.add(idx_n)

        code_diagram = self.create_code_diagram(
            "Sampling",
            "Clustering",
            "Connecting",
            "",
            "Smoothing",
            "",
            "Tetrahedralizing",
            "Prolonging",
            z_index=80
        )
        code_diagram.move_to(ORIGIN).to_edge(LEFT).shift(0.2 * DOWN)
        code_diagram.shift(LEFT * 4)
        self.play(
            code_diagram.animate.shift(RIGHT * 4),
            run_time=0.6,
            rate_func=rush_from
        )
        self.pause("Show code diagram")

        sample_scale = 1.35
        nonsample_scale = 0.8
        self.play(
            code_diagram[0][0][0][0].animate.set_stroke(C_ORANGE).set_fill(lerp(C_WHITE, C_ORANGE, 0.5)),
            *[obj.animate.set_fill(lerp(C_WHITE, obj.get_fill_color(), 0.5)).set_stroke(opacity=0).scale(nonsample_scale) for obj in vertices_fine],
            *[obj.animate.scale(sample_scale) for obj in vertices_coarse],
            run_time=0.4
        )

        debug_indices = []
        if not FINAL:
            for i in range(len(vertices_coarse)):
                debug_index = Text(f"{i}", color=C_RED)
                debug_index.move_to(vertices_coarse[i])
                debug_index.set_z_index(1000)
                debug_indices.append(debug_index)
        self.add(*debug_indices)
        self.pause("Sample vertices")
        self.remove(*debug_indices)

        conn_r = 2.4
        to_connect = set()
        for i in range(len(vertices_coarse)):
            v_i = vertices_coarse[i].get_center()
            for j in range(i + 1, len(vertices_coarse)):
                v_j = vertices_coarse[j].get_center()

                if np.linalg.norm(v_j - v_i) < conn_r:
                    to_connect.add((i, j))

        to_connect = to_connect \
            - {(4, 9)} \
            | {(2, 9), (1, 10), (0, 6), (3, 5)}

        edges_coarse = []
        edges_coarse_boundary = []
        for i, j in to_connect:
            if np.random.rand() < 0.5:
                i, j = j, i
            v_i, v_j = vertices_coarse[i].get_center(), vertices_coarse[j].get_center()
            edge_coarse = Line(v_i, v_j).set_stroke(lerp(C_LIGHT_GRAY, C_GRAY, 0.5), width=16).set_cap_style(CapStyleType.ROUND)
            edge_coarse.start_index = i
            edge_coarse.end_index = j
            edge_coarse.set_z_index(24)
            edges_coarse.append(edge_coarse)

            if boundary[samples[i]] and boundary[samples[j]]:
                edges_coarse_boundary.append(edge_coarse)

        in_cluster = [-1] * len(vertices_fine)
        for i_f in range(len(vertices_fine)):
            dist_best = 1e10
            for j_c, j_f in enumerate(samples):
                if boundary[i_f] > boundary[j_f]:
                    continue
                dist = np.linalg.norm(vertices_fine[i_f].get_center() - vertices_fine[j_f].get_center())
                if dist < dist_best:
                    in_cluster[i_f] = j_c
                    dist_best = dist

        clusters = [[] for _ in range(max(in_cluster) + 1)]
        for idx, c in enumerate(in_cluster):
            clusters[c].append(idx)

        cluster_polygons = []
        for i, cluster in enumerate(clusters):
            color = lerp(C_WHITE, boundary_colors[boundary[samples[i]]], 0.25)

            polygons = Group()
            drawn = set()
            for a in cluster:
                if len(cluster) == 1:
                    dot = Circle(0.0).set_stroke(color, width=64).set_fill(color, opacity=1).set_cap_style(CapStyleType.ROUND)
                    dot.move_to(vertices_fine[a].get_center())
                    polygons.add(dot)

                for b in [j for j in cluster if j in neighf[a]]:
                    va, vb = [vertices_fine[j].get_center() for j in (a, b)]
                    key = tuple(sorted([a, b]))
                    if len(set(key)) == len(key) and key not in drawn:
                        drawn.add(key)

                        line = Line(va, vb).set_stroke(color, width=64).set_fill(color, opacity=1).set_cap_style(CapStyleType.ROUND)
                        polygons.add(line)

                    for c in [j for j in cluster if j in neighf[a] & neighf[b]]:
                        key = tuple(sorted([a, b, c]))
                        if len(set(key)) == len(key) and key not in drawn:
                            drawn.add(key)

                            vc = vertices_fine[c].get_center()
                            eab, ebc = vb - va, vc - vb
                            eab /= np.linalg.norm(eab)
                            ebc /= np.linalg.norm(ebc)
                            if np.linalg.norm(np.cross(eab, ebc)) < 1e-1:
                                continue
                            polygon = Polygon(va, vb, vc).set_stroke(color, width=64).set_fill(color, opacity=1).set_cap_style(CapStyleType.ROUND).round_corners(0.05)
                            polygons.add(polygon)

            for obj in polygons:
                obj.generate_target()
                obj.set_stroke(C_WHITE).set_fill(C_WHITE)
            cluster_polygons.append(polygons)

        connecting_edges = {}
        for edge in edges_fine:
            key = tuple(sorted(in_cluster[idx] for idx in [edge.start_index, edge.end_index]))
            if key[0] != key[1]:
                if key not in connecting_edges:
                    connecting_edges[key] = []
                connecting_edges[key].append(edge)

        nonconnecting_edges = {*edges_fine}
        for edges in connecting_edges.values():
            nonconnecting_edges -= {*edges}

        self.play(
            code_diagram[0][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            code_diagram[1][0][0][0].animate.set_stroke(C_ORANGE).set_fill(lerp(C_WHITE, C_ORANGE, 0.5)),
            *[MoveToTarget(obj) for group in cluster_polygons for obj in group],
            *[FadeOut(obj) for obj in nonconnecting_edges],
            run_time=0.4
        )
        debug_indices = []
        if not FINAL:
            for i in range(len(vertices_fine)):
                debug_index = Text(f"{in_cluster[i]}", color=C_RED)
                debug_index.move_to(vertices_fine[i])
                debug_index.set_z_index(1000)
                debug_indices.append(debug_index)
        self.add(*debug_indices)
        self.pause("Draw clusters")
        self.remove(*debug_indices)

        self.play(
            code_diagram[1][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            code_diagram[2][0][0][0].animate.set_stroke(C_ORANGE).set_fill(lerp(C_WHITE, C_ORANGE, 0.5)),
            *[obj.animate.set_stroke(opacity=0) for v in connecting_edges.values() for obj in v],
            *[Create(obj) for obj in edges_coarse],
            run_time=0.6
        )
        self.pause("Draw coarse edges")

        self.play(
            code_diagram[2][0][0][0].animate.set_stroke(C_LIGHT_GRAY).set_fill(lerp(C_WHITE, C_LIGHT_GRAY, 0.5)),
            code_diagram[3][0][0][0].animate.set_stroke(C_ORANGE).set_fill(lerp(C_WHITE, C_ORANGE, 0.5)),
            run_time=0.6
        )
        self.pause("Highlight Smoothing step")

        for it in range(2):
            to_restore = [
                *vertices_coarse,
                *edges_coarse,
            ]
            for obj in to_restore:
                obj.save_state()

            if it == 1:
                to_shift = [
                    *vertices_fine,
                    *vertices_coarse,
                    *edges_coarse,
                    *cluster_polygons
                ]
                shift = 2.4 * LEFT
                self.play(
                    FadeOut(code_diagram, shift=shift),
                    *[obj.animate.shift(shift) for obj in to_shift],
                    run_time=0.6
                )

                spacing_vert = 1.25
                spacing_hori = 1.25

                clusters_small = []
                vertices_small = []
                molds_small = []
                molds_to_delete_small = []
                vertices_to_delete_small = []
                for iy in range(3):
                    for ix in range(3):
                        color = lerp(C_WHITE, boundary_colors[iy], 0.35)
                        cluster_small = Circle(0.0).set_stroke(color, width=96).set_fill(color, opacity=1)
                        cluster_small.move_to(np.array([spacing_hori * -ix, spacing_vert * iy, 0.0]))
                        cluster_small.set_z_index(4)
                        clusters_small.append(cluster_small)

                        color = lerp(C_WHITE, boundary_colors[ix], 0.65)
                        vertex_small = vertices_fine[0].copy().set_fill(color).set_stroke(opacity=0)
                        vertex_small.move_to(cluster_small)
                        vertex_small.set_z_index(5)
                        vertices_small.append(vertex_small)

                        mold_small = vertex_small.copy().set_fill(opacity=0).set_stroke(color, opacity=1, width=8).scale(1.25 * sample_scale / nonsample_scale)
                        mold_small.move_to(cluster_small)
                        mold_small.set_z_index(5)
                        molds_small.append(mold_small)

                        if ix < iy:
                            vertices_to_delete_small.append(vertex_small)
                            molds_to_delete_small.append(mold_small)

                cases_group = Group(*clusters_small, *vertices_small, *molds_small)
                cases_group.move_to(np.array([3.6, -0.2, 0.0]))
                self.play(
                    FadeIn(cases_group, scale=0.8),
                    run_time=0.6
                )
                self.pause("Show all cases")

                scale = 1.5
                self.play(
                    *[obj.animate.scale(scale).set_stroke(C_RED) for obj in molds_to_delete_small],
                    run_time=0.4
                )
                self.hold(0.8)
                self.play(
                    *[obj.animate.scale(1 / scale) for obj in molds_to_delete_small],
                    run_time=0.4
                )
                self.pause("Highlight small molds to remove")

                self.play(
                    *[FadeOut(obj) for obj in molds_to_delete_small],
                    *[FadeOut(obj) for obj in vertices_to_delete_small],
                    run_time=0.6
                )
                self.pause("Remove small some molds")

            Vc_new = [np.zeros((3,)) for _ in range(len(samples))]
            Vc_new_num = [0] * len(samples)
            for idx, vertex in enumerate(vertices_fine):
                clu = in_cluster[idx]
                if it and boundary[idx] < boundary[samples[clu]]:
                    continue
                Vc_new[clu] += vertex.get_center()
                Vc_new_num[clu] += 1
            Vc_new = np.array([p / n for p, n in zip(Vc_new, Vc_new_num)])

            molds = []
            molds_to_delete = []
            vertices_to_delete = []
            for idx, vertex_fine in enumerate(vertices_fine):
                clu = in_cluster[idx]
                vertex_coarse = vertices_coarse[clu]
                color = lerp(C_WHITE, boundary_colors[boundary[idx]], 0.65)

                vertex_fine.generate_target()
                vertex_fine.target.set_fill(color)

                mold = vertex_fine.copy().set_fill(opacity=0).set_stroke(color, opacity=1, width=8)
                mold.scale(1.25 * sample_scale / nonsample_scale)
                mold.set_z_index(50)
                molds.append(mold)

                mold.generate_target()
                mold.target.set_stroke(boundary_colors[boundary[samples[clu]]])
                mold.target.move_to(Vc_new[clu])
 
                if it and boundary[idx] < boundary[samples[clu]]:
                    mold.set_z_index(51)
                    molds_to_delete.append(mold)
                    vertices_to_delete.append(vertex_fine)

            self.play(
                *[MoveToTarget(obj) for obj in vertices_fine],
                *[FadeIn(obj, scale=0) for obj in molds],
                run_time=0.6
            )
            self.pause("Highlight fine points for averaging")

            if molds_to_delete:
                scale = 1.5
                self.play(
                    *[obj.animate.scale(scale).set_stroke(C_RED) for obj in molds_to_delete],
                    run_time=0.4
                )
                self.hold(0.8)
                self.play(
                    *[obj.animate.scale(1 / scale) for obj in molds_to_delete],
                    run_time=0.4
                )
                self.pause("Highlight molds to remove")

                self.play(
                    *[FadeOut(obj) for obj in molds_to_delete],
                    *[FadeOut(obj) for obj in vertices_to_delete],
                    run_time=0.6
                )
                self.pause("Remove some molds")

                molds = [obj for obj in molds if obj not in molds_to_delete]

            self.play(
                *[MoveToTarget(obj) for obj in molds],
                run_time=0.8
            )
            if it == 0:
                self.pause("Average fine cluster points")
            else:
                self.hold(0.2)

            self.play(
                *[obj.animate.move_to(Vc_new[idx]) for idx, obj in enumerate(vertices_coarse)],
                *[obj.animate.put_start_and_end_on(Vc_new[obj.start_index], Vc_new[obj.end_index]) for obj in edges_coarse],
                run_time=0.8
            )
            self.play(
                *[FadeOut(obj) for obj in molds],
                run_time=0.4
            )
            if it == 0:
                self.pause("Move coarse vertices")
            else:
                self.hold(0.2)

            edges_coarse_boundary = sorted(edges_coarse_boundary, key=lambda e: np.atan2(*e.get_center()[:2]))
            self.play(
                AnimationGroup(
                    *[obj.animate.set_stroke(C_DARK_GRAY, width=24) for obj in edges_coarse_boundary],
                    lag_ratio=0.3
                ),
                run_time=0.8
            )
            self.pause("Highlight boundary")

            if it == 0:
                self.play(
                    *[obj.animate.restore() for obj in to_restore],
                    run_time=1.2
                )
                self.pause("Restore state")

        mg_positions = [np.array([4.8 * (min(1, j) - 0.5), -0.6 + 1.5 * ((j + 1) % 3 - 1), 0.0]) for j in range(3)]
        mg_images = [None] * 3
        mg_arrows = [None] * 2
        for i, name in enumerate(["0", "all_1", "none_1"]):
            image = self.load_image(f"cube_mg_smoothing_{name}")
            image.scale_to_fit_height(2.8)
            image.move_to(mg_positions[i])
            image.set_z_index(60)
            mg_images[i] = image

            if i:
                start = mg_positions[0]
                diff = mg_positions[i] - start
                arrow = Arrow(color=C_DARK_GRAY).put_start_and_end_on(start + 0.35 * diff, start + 0.65 * diff)
                arrow.set_z_index(60)
                mg_arrows[i - 1] = arrow

        white_rectangle = Rectangle(C_WHITE, SIZE[1] + 1, SIZE[0] + 1).set_fill(opacity=0.85)
        white_rectangle.set_z_index(50)
        self.play(
            FadeIn(white_rectangle),
            *[FadeIn(obj, shift=UP * 0.5) for obj in [*mg_images[:2], mg_arrows[0]]],
            run_time=0.6
        )
        self.pause("Show all smoothing MG")

        self.play(
            *[FadeIn(obj, shift=UP * 0.5) for obj in [mg_images[2]]],
            Transform(mg_arrows[0], mg_arrows[1]),
            run_time=0.6
        )
        self.pause("Show boundary-aware smoothing MG")

        self.clear(run_time=0.6)

    def animate_slide_experiment_revised(self):
        self.pause()

        buenos = [self.load_image("bueno") for _ in range(3)]
        for idx, obj in enumerate(buenos):
            obj.scale_to_fit_height(1e-3)
            obj.set_z_index(80 + idx)
            obj.move_to(np.array([2.4 * (2 * idx % 3 - 1), 0.4 * (1 - 2 * (idx // 2)), 0.0]))
        self.play(
            *[obj.animate.scale_to_fit_height(5.0) for obj in buenos],
            run_time=0.8,
            rate_func=linear
        )
        self.pause("Buenos!")

        self.clear(run_time=0.6)

    def animate_slide_summary_conclusion(self):
        pass

    def animate_slide_last_page(self):
        # TODO: replace with something?
        self.animate_slide_first_page()

    ###################################
    #                                 #
    #            STRUCTURE            #
    #                                 #
    ###################################

    def animate(self):
        self.animate_slide_first_page()

        self.animate_slide_problem_summary() # TODO

        self.animate_slide_dirichlet_demonstration()
        self.animate_slide_multigrid_diagram()
        self.animate_slide_prolongation_demonstration()
        self.animate_slide_gravo_demonstration()
        self.animate_slide_tetrahedral_meshes()

        self.animate_slide_contribution_basics()
        self.animate_slide_experiments() # TODO
        self.animate_slide_boundary() # TODO
        self.animate_slide_optimization_1_vertex_ordering()
        self.animate_slide_optimization_2_sampling_density() # TODO
        self.animate_slide_optimization_3_pit_prevention()
        self.animate_slide_optimization_4_boundary_aware_smoothing()
        self.animate_slide_experiment_revised() # TODO

        self.animate_slide_summary_conclusion() # TODO
        self.animate_slide_last_page() # TODO

if __name__ == "__main__":
    exit_code = render_slides()

    if exit_code == 0:
        if CONVERT_AFTER:
            from bb_convert import convert
            convert()

        if PRESENT_AFTER:
            from ba_present import present
            present(
                FINAL_FRAMERATE if FINAL else DEBUG_FRAMERATE,
                FINAL,
                True
            )
