import heapq
from manim import *
import numpy as np
import os
import scipy.sparse as sp
import shutil
import time

from x_utils import *

DEBUG = False
FROM = 0
TO = 10000

BACKGROUND_COLOR = C_WHITE
SIZE = (8.0 * 16 / 9, 8.0)

RENDER_PATH = f"{PATH}\\media\\images\\a_render"

config.background_color = BACKGROUND_COLOR
config.max_files_cached = 1000

def render_slides():
    start_time = time.time()

    if os.path.exists(RENDER_PATH):
        shutil.rmtree(RENDER_PATH)

    width, height = DEBUG_SIZE if DEBUG else DEFAULT_SIZE
    framerate = DEBUG_FRAMERATE if DEBUG else DEFAULT_FRAMERATE

    filename = os.path.realpath(__file__)
    command = f"manim {filename} PresentationScene --resolution {width},{height} --frame_rate {framerate} --format=png --disable_caching --from_animation_number {FROM},{TO}"

    print(f"\033[0;32m{command}\033[0m")
    exit_code = os.system(command)

    if exit_code != 0:
        return exit_code

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)

    print("\033[34;1mCopying frames...\033[0m")

    for filename in os.listdir(RENDER_PATH):
        nr = int("".join(c for c in filename if c.isdigit()))
        shutil.copyfile(f"{RENDER_PATH}/{filename}", f"{OUTPUT_PATH}/{nr:06}.png")

    duration = int(time.time() - start_time)
    print(f"\033[32;1mFinished in {duration // 60}m {duration % 60:02}s!\033[0m")

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

    def update_page(self):
        if self.page_number_text:
            self.remove(self.page_number_text)
            self.remove(self.page_number_background)

        self.page_number += 1

        self.page_number_background = always_redraw(
            lambda: Rectangle(C_WHITE, 0.4, 0.6).round_corners(0.1).set_fill(opacity=1).set_stroke(C_WHITE, opacity=0.5, width=6 * self.camera.frame_height / 8.0) \
                .to_corner(DOWN + RIGHT).scale(self.camera.frame_height / 8.0, about_point=ORIGIN).shift(self.camera.frame_center) \
                .set_z_index(101)
        )
        self.page_number_text = always_redraw(
            lambda: Text(str(self.page_number), color=C_DARK_GRAY).set_stroke(opacity=0) \
                .scale(0.4 * self.camera.frame_height / 8.0).move_to(self.page_number_background)
                .set_z_index(102)
        )

        self.add(self.page_number_background)
        self.add(self.page_number_text)

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

    def load_image(self, name):
        image = ImageMobject(f"{PATH}/assets/{name}.png")
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
        return image

    def clear(self, fade=0.0):
        if fade:
            white_rectangle = Rectangle(WHITE, 100, 100).set_fill(opacity=1)
            white_rectangle.set_z_index(99)
            self.play(
                FadeIn(white_rectangle),
                run_time=fade
            )

        self.remove(*self.all_objects())
        self.camera.frame.move_to(ORIGIN)
        self.camera.frame.scale_to_fit_height(SIZE[1])

    def set_title(self, text, **kwargs):
        kwargs["color"] = kwargs.get("color", C_BLACK)

        self.title = Text(text, **kwargs).scale(0.8).to_corner(UP + LEFT).shift((0.2, -0.3, 0))
        self.add(self.title)

        self.next_bullet_point_pos = self.title.get_corner(UP + LEFT) + DOWN * 0.9

    def add_bullet_point(self, text, **kwargs):
        kwargs["color"] = kwargs.get("color", "#1F1F1F")

        spacing = len(text) - len(text.strip())
        text = text.strip()

        last_bullet_point = Text(text, **kwargs).scale(0.6).move_to(self.next_bullet_point_pos, LEFT + UP).shift(0.3 * spacing * RIGHT)
        self.add(last_bullet_point)

        self.next_bullet_point_pos += DOWN * 0.6

        return last_bullet_point

    def create_arrow(self, arrow, opacity=1):
        start, end = arrow.get_start_and_end()
        opacity = arrow.get_opacity()
        arrow.set_opacity(0).put_start_and_end_on(start, start + 0.00001 * (end - start))
        return arrow.animate.set_opacity(opacity).put_start_and_end_on(start, end)

    def construct(self):
        self.page_number = 0

        self.title = None
        self.page_number_text = None
        self.update_page()

        self.animate()

    ################################
    #                              #
    #            SLIDES            #
    #                              #
    ################################

    def animate_slide_intro_outro(self):
        paper_title_text = Text("A Fast Geometric Multigrid Method\nfor Volumetric Meshes", color=C_BLACK).scale(1.1).move_to(ORIGIN).shift(DOWN * 0.9)
        authors_text = Text("Tim Huisman", color=DARK_GREY).scale(0.6).next_to(paper_title_text, DOWN)
        presented_by_text = Text("Presented by Tim Huisman", color=GREY).scale(0.6).next_to(authors_text, DOWN).shift(DOWN * 0.4)

        self.add(paper_title_text, authors_text, presented_by_text)

        self.pause("Intro")

        self.clear()

    def animate_slide_dirichlet_demonstration(self):
        self.pause("Start dirichlet_demonstration")

        n = 16
        size = 6
        thickness = 0.1

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

                cell = Square(0.96 * spacing).round_corners(0.05).set_stroke(width=0).set_fill(C_LIGHT_GRAY, opacity=1)
                cell.move_to((x, y, 0))
                cells[iy][ix] = cell

                cell_hot = Square(0.96 * spacing).round_corners(0.05).set_stroke(width=0).set_fill(C_ORANGE, opacity=1).set_opacity(temperature)
                cell_hot.move_to((x, y, 0))
                cells_hot[iy][ix] = cell_hot

                cell_cold = Square(0.96 * spacing).round_corners(0.05).set_stroke(width=0).set_fill(C_BLUE, opacity=1).set_opacity(-temperature)
                cell_cold.move_to((x, y, 0))
                cells_cold[iy][ix] = cell_cold

                question_mark = Tex("$\\textbf{?}$", color=C_DARK_GRAY)
                question_mark.scale(0.8).move_to(cell)
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
        self.play(
            *[FadeOut(obj) for obj in to_zoom_disappear],
            FadeIn(question_marks[fy][fx], scale=0),
            self.camera.frame.animate.scale(1 / camera_zoom_1).shift(cells[fy][fx + 1].get_center()),
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
        cells_hot[fy][fx].set_opacity(temperature_avg)
        cells_cold[fy][fx].set_opacity(-temperature_avg)
        to_move = []
        to_keep_in_front = []
        for d in range(4):
            nx = fx - (d == 0) + (d == 1)
            ny = fy - (d == 2) + (d == 3)

            to_keep_in_front.append(cells[ny][nx])
            to_keep_in_front.append(cells_hot[ny][nx])
            to_keep_in_front.append(temperature_texts[d])

            cell_hot_copy = cells_hot[ny][nx].copy()
            cell_hot_copy.generate_target()
            cell_hot_copy.target.move_to(cells[fy][fx])
            cell_hot_copy.target.set_opacity(0)
            to_move.append(cell_hot_copy)

        self.add_foreground_mobjects(*to_keep_in_front)
        self.play(
            *[MoveToTarget(obj) for obj in to_move],
            FadeIn(cells_hot[fy][fx]),
            FadeIn(cells_cold[fy][fx]),
            FadeOut(question_marks[fy][fx], scale=0),
            FadeIn(temperature_texts[4], scale=0),
            run_time=0.8
        )
        self.remove_foreground_mobjects(*to_keep_in_front)
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

        system_container = Rectangle(C_GRAY, 1.2, 0.8).set_stroke(width=1.2).set_fill(C_WHITE, opacity=1)
        system_container.move_to(cells[fy][fx + 1]).shift((1.2, 0.2, 0.0))
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
        amount = 48 if not DEBUG else 8
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

        camera_shift_2 = RIGHT * 2

        system_group = Group(system_container, *equations_tex)
        for row in equations_cells:
            system_group.add(*row)
        for obj in system_group:
            obj.generate_target()
            obj.target.shift(-cells[fy][fx + 1].get_center()).scale(camera_zoom_1, about_point=ORIGIN).shift(camera_shift_2)
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
            self.camera.frame.animate.shift(-cells[fy][fx + 1].get_center()).scale(camera_zoom_1).shift(camera_shift_2),
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
            system_group.animate.shift(DOWN * 0.8),
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

        self.play(
            *[FadeOut(obj, scale=0) for obj in question_marks_flat],
            run_time=0.6
        )
        self.pause("Initialize all at zero")

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

        states = [
            np.zeros((n + 2, n + 2), dtype=float)
        ]
        iterations = 100
        iterations_to_show = [1, 2, 5, 20, 100]
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
                for obj in objs:
                    to_keep_in_front.append(obj)
                    obj_moving = obj.copy()
                    obj_moving.set_stroke(opacity=0)
                    obj_moving.generate_target()
                    obj_moving.target.move_to(cells[fy][fx])
                    obj_moving.target.set_opacity(0)
                    to_move.append(obj_moving)

            self.add_foreground_mobjects(*to_keep_in_front)
            self.play(
                *[MoveToTarget(obj) for obj in to_move],
                cells_hot[fy][fx].animate.set_opacity(states[1][fy][fx]),
                cells_cold[fy][fx].animate.set_opacity(-states[1][fy][fx]),
                run_time=0.8 if idx < 2 else 0.6
            )
            self.remove_foreground_mobjects(*to_keep_in_front)
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

        camera_shift_4 = UP * 0.12
        self.play(
            first_gs_animation,
            self.camera.frame.animate.shift((amount - 1) * LEFT * spacing).shift(-camera_shift_3).scale(camera_zoom_3).shift(camera_shift_4),
            FadeOut(cells_boundary[1][amount - 1]),
            run_time=1.6
        )
        self.pause("Put camera back again")

        iteration_counter_text = None
        for it in iterations_to_show:
            iteration_counter_text_new = MathTex(f"{it}", "\\text{ iteration}", "\\text{s}", color=C_DARK_GRAY)
            if it == 1:
                iteration_counter_text_new[-1].set_color(C_WHITE)
            iteration_counter_text_new.scale(0.8).next_to(walls[3], UP).shift(DOWN * 0.04)
            text_animations = []
            if iteration_counter_text is None:
                text_animations.append(FadeIn(iteration_counter_text_new, scale=0.8))
            else:
                text_animations.append(TransformMatchingTex(iteration_counter_text, iteration_counter_text_new))

            color_animations = ([
                cells_hot[jy][jx].animate.set_opacity(states[it][jy][jx])
                for jy in range(n) for jx in range(n)
            ] + [
                cells_cold[jy][jx].animate.set_opacity(-states[it][jy][jx])
                for jy in range(n) for jx in range(n)
            ]) * (not DEBUG)

            self.play(
                *text_animations,
                *color_animations,
                run_time=0.4
            )
            self.pause(f"Show state {it}")

            iteration_counter_text = iteration_counter_text_new

        self.clear(fade=0.6)

    def animate_slide_multigrid_diagram(self):
        pass

    def animate_slide_gravo_demonstration(self):
        self.pause("Start gravo_demonstration")

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
            edge = Line(V[E[i][0]], V[E[i][1]]).set_stroke(light_blue, width=8)
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
        edges_lose = edges_fine[-lose:]
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

        self.pause("Step 1, sampling")

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

        self.pause("Step 2, clustering")

        q = [(0.0, 0, idx, -1, j) for j, idx in enumerate(samples)]
        pred = [None] * Vf.shape[0]
        cluster = [None] * Vf.shape[0]
        depth = [None] * Vf.shape[0]
        Ec = set()
        while q:
            c, d, cur, prv, idx = heapq.heappop(q)

            if cluster[cur] is not None:
                x, y = cluster[cur], cluster[prv]
                if x != y:
                    Ec.add((x, y))
                continue
            cluster[cur] = idx
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

        colors = [color_map[c] for c in color_indices]

        self.play(
            *[obj.animate.set_color(colors[j]).set_stroke(C_WHITE, width=4, opacity=1) for j, obj in enumerate(vertices_coarse)]
        )
        self.pause("Color points")

        cluster_edges = []
        dijkstra_animations = [([], []) for _ in range(10)]
        for cur in range(Vf.shape[0]):
            prv = pred[cur]
            d = depth[cur]
            clu = cluster[cur]

            if d:
                edge = Line(Vf[prv], Vf[cur]).set_stroke(colors[clu], width=12)
                edge.set_z_index(15)
                edge.cluster = clu
                cluster_edges.append(edge)
                dijkstra_animations[d][0].append(
                    Create(edge)
                )
                dijkstra_animations[d][1].append(
                    vertices_fine[cur].animate.set_fill(colors[clu]).set_stroke(C_WHITE, width=4, opacity=1).scale(1.25)
                )
            else:
                vertices_fine[cur].set_fill(colors[clu]).set_stroke(C_WHITE, width=4, opacity=1).scale(1.25)
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

        cluster_edges_new = []
        connecting_edges = {}
        for idx, (a, b) in enumerate(Ef):
            a, b = int(a), int(b)
            p, q = cluster[a], cluster[b]
            if p != q:
                key = tuple(sorted([p, q]))
                if key not in connecting_edges:
                    connecting_edges[key] = []
                connecting_edges[key].append(edges_fine[idx])
                continue
            if pred[a] == b or pred[b] == a:
                continue

            edge = Line(Vf[a], Vf[b]).set_stroke(colors[p], width=12)
            edge.cluster = p
            edge.set_z_index(20)
            cluster_edges_new.append(edge)
        cluster_edges.extend(cluster_edges_new)
        self.play(
            *[FadeIn(obj, scale=0) for obj in cluster_edges_new],
            run_time=0.4
        )
        self.pause("Add all cluster edges")

        self.play(
            *[obj.animate.set_fill(lerp(C_WHITE, str(obj.get_color()), 0.35)) for obj in vertices_fine],
            *[obj.animate.set_stroke(lerp(C_WHITE, str(obj.get_color()), 0.35)) for obj in cluster_edges],
            run_time=0.4
        )
        self.pause("Fade fine part of the clusters a bit")

        min_key = min(connecting_edges.keys(), key=lambda x: np.linalg.norm(Vc[x[0]] + Vc[x[1]]))
        highlight_vertex_color_pairs = []
        for idx, vertex in enumerate(vertices_fine):
            if cluster[idx] in min_key:
                highlight_vertex_color_pairs.append((vertex, color_indices[cluster[idx]]))
        highlight_edge_color_pairs = []
        for idx, edge in enumerate(cluster_edges):
            if edge.cluster in min_key:
                highlight_edge_color_pairs.append((edge, color_indices[edge.cluster]))
        self.play(
            *[obj.animate.set_fill(color_map[c]) for obj, c in highlight_vertex_color_pairs],
            *[obj.animate.set_stroke(color_map[c]) for obj, c in highlight_edge_color_pairs],
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
                obj.animate.put_start_and_end_on(Vc[cluster[obj.start_index]], Vc[cluster[obj.end_index]]) for obj in edges
            ]
            edge_coarse = Line(Vc[a], Vc[b]).set_stroke(C_GRAY, width=12)
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
            *[obj.animate.set_fill(lerp(C_WHITE, color_map[c], 0.35)) for obj, c in highlight_vertex_color_pairs],
            *[obj.animate.set_stroke(lerp(C_WHITE, color_map[c], 0.35)) for obj, c in highlight_edge_color_pairs],
            run_time=0.4
        )
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
        self.play(
            *[FadeOut(obj) for obj in edges_fine],
            run_time=0.2
        )
        self.pause("Draw all coarse edges")

        self.pause("Step 3, smoothing")

        Vc_new = [np.zeros((3,)) for _ in range(len(samples))]
        Vc_new_num = [0] * len(samples)
        for idx, vertex in enumerate(vertices_fine):
            clu = cluster[idx]
            Vc_new[clu] += Vf[idx]
            Vc_new_num[clu] += 1
        Vc_new = np.array([p / n for p, n in zip(Vc_new, Vc_new_num)])

        molds = []
        for idx, vertex_fine in enumerate(vertices_fine):
            clu = cluster[idx]
            vertex_coarse = vertices_coarse[clu]

            mold = vertex_fine.copy().set_fill(opacity=0).set_stroke(vertex_fine.get_fill_color(), width=8)
            mold.scale(1.25 * vertex_coarse.width / vertex_fine.width)
            mold.set_z_index(50)
            molds.append(mold)

            mold.generate_target()
            mold.target.set_stroke(colors[clu])
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

        self.pause("Step 4, triangulating")

        self.play(
            *[FadeOut(obj) for obj in cluster_edges],
            *[obj.animate.set_fill(lerp(C_WHITE, C_BLUE, 0.35)) for obj in vertices_fine],
            *[obj.animate.set_fill(C_GREEN) for obj in vertices_coarse],
            run_time=0.6
        )
        self.pause("Recolor points, fade cluster edges")

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
        self.pause("Focus on fine vertex to prolongate")

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

        colors = [C_RED, C_ORANGE, C_PURPLE]
        light_colors = [lerp(C_WHITE, color, 0.35) for color in colors]
        self.play(
            *[vertices_coarse[idx].animate.set_fill(colors[j]) for j, idx in enumerate(Tc[0])],
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
                Line(Vf[vf_idx], Vc[idx]).set_sheen_direction(Vc[idx] - Vf[vf_idx]).set_stroke([C_BLUE, C_BLUE] + 2 * [vertices_coarse[idx].get_fill_color()], width=6, opacity=0.5) \
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

        self.clear(fade=0.6)

    ###################################
    #                                 #
    #            STRUCTURE            #
    #                                 #
    ###################################

    def animate(self):
        self.animate_slide_intro_outro()

        self.animate_slide_dirichlet_demonstration()

        self.animate_slide_multigrid_diagram()

        self.animate_slide_gravo_demonstration()

if __name__ == "__main__":
    exit_code = render_slides()

    if exit_code == 0:
        from ba_present import present
        present(
            DEBUG_FRAMERATE if DEBUG else DEFAULT_FRAMERATE,
            False,
            True
        )
