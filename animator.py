import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models import Point

class Animator():
    def __init__(self, points: Point, expected_range: int = 500):
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

    def _animate(self, _frame_i):
        offsets = [point.position for point in self._points]
        self.scatter_plot.set_offsets(offsets)

        for text, offset in zip(self._texts, offsets):
            text.set_position(offset)

        return (self._ax, self.scatter_plot)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points

    def show_animation(self):
        anim = animation.FuncAnimation(
            self._fig,
            self._animate,
            init_func=self._init_animation,
            interval=20,
            blit=True,
            save_count=50
        )
        plt.show()

def plot_points(points):
    xs = [point.position[0] for point in points]
    ys = [point.position[1] for point in points]
    plt.scatter(xs, ys, s=100)
    for point in points:
        plt.text(*point.position, point.name)

    plt.show()
