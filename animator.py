from typing import List

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from models import Point

class Animator():
    def __init__(self, points: List[Point], expected_range: int = 0.5):
        self._points = points
        self._expected_range = expected_range

        self._fig, self._ax = plt.subplots()

    def _init_animation(self):
        self._ax.set_xlim(-1 * self._expected_range, self._expected_range)
        self._ax.set_ylim(-1 * self._expected_range, self._expected_range)

        self.scatter_plot = self._ax.scatter([], [], s=100)

        self._texts = [
            self._ax.annotate(point.name, point.position)
            for point in self._points
        ]

        return (self._ax, self.scatter_plot)

    def _animate(self, _frame_i: int):
        offsets = [point.position for point in self._points]
        self.scatter_plot.set_offsets(offsets)

        for text, offset in zip(self._texts, offsets):
            text.set_position(offset)

        return (self._ax, self.scatter_plot)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points: List[Point]):
        self._points = points

    def show_animation(self):
        animation = FuncAnimation(
            self._fig,
            self._animate,
            init_func=self._init_animation,
            interval=20,
            blit=True,
            save_count=50
        )
        plt.show()

        #animation.save("optimization.gif", dpi=300, writer=PillowWriter(fps=25))
