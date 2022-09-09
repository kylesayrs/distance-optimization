from typing import List, Optional

import argparse
import numpy
import threading

import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
args = {
    "minimum_loss": 0.0,
    "max_steps": 1000,
    "learning_rate": 0.5,
    "verbose": True,
    "animate": True,
}
"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--minimum_loss", default=0.0)
parser.add_argument("--max_steps", default=1000)
parser.add_argument("--learning_rate", default=0.5)
parser.add_argument("--verbose", default=False)
parser.add_argument("--animate", default=True)

class Point():
    def __init__(
        self,
        target_distances: List[float] = [],
        position: List[float] = None,
        name: str = "",
        num_dims: int = 2,
        init_scale: float = 500,
    ):
        self.target_distances = numpy.array(target_distances)
        self.name = name
        self._num_dims = num_dims
        self._init_scale = init_scale

        if position:
            self.position = numpy.array(position, dtype=numpy.float32)
        else:
            self.position = self.initialize_position()

    def initialize_position(self):
        position = numpy.random.normal(
            loc=0.0,
            scale=self._init_scale,
            size=(self._num_dims, )
        )
        self.position = position
        return position

    def __repr__(self):
        return str(self)

    def __str__(self):
        positions_string = ", ".join([f"{pos:0.2f}" for pos in self.position])
        if self.name:
            return f"Point(name=\"{self.name}\", ({positions_string}))"
        else:
            return f"Point(({positions_string}))"

class MSELoss():
    def __init__(self, points):
        self._points = points

    def calc_total_loss(self, weighted=False):
        weights=[
            numpy.count_nonzero(point.target_distances is not None)
            for point in self._points
        ] if weighted else None

        return numpy.average(self.calc_point_losses(), weights=weights)

    def calc_point_losses(self):
        return [self.calc_loss(point) for point in self._points]

    def calc_loss(self, point) -> float:
        losses = []
        for target_point, target_distance in zip(self._points, point.target_distances):
            if target_distance is None: continue

            actual_distance = numpy.linalg.norm(point.position - target_point.position)

            loss = (actual_distance - target_distance) ** 2
            losses.append(loss)

        if losses:
            return numpy.mean(losses)
        else:
            return 0.0

    def calc_gradient(self, point: Point) -> numpy.ndarray:
        """
        E = (D - t) ^ 2
        dE/dx = 2(D - t) * (dD/dx)
        dD/dx = (1/d)(x2 - x1)

        dE/dx = 2 * ((D - t) / D) * (x2 - x1)

        :param point: point whose gradient is being calculated
        :return: gradient wrt MSE loss
        """
        point_positions = [_point.position for _point in self._points]
        target_gradients = []
        for target_point, target_distance in zip(self._points, point.target_distances):
            if target_distance is None: continue

            actual_distance = numpy.linalg.norm(point.position - target_point.position)

            target_gradient = (actual_distance - target_distance) / actual_distance * (point.position - target_point.position)
            target_gradients.append(target_gradient)

        return numpy.average(target_gradients, axis=0)

class SimpleOptimizer():
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate
        self.total_steps = 0

    def step(self, point, gradient):
        point.position -= gradient * self._learning_rate
        self.total_steps += 1

def numpy_softmax(x: numpy.ndarray, axis: int = 0):
    """
    :param x: array containing values to be softmaxed
    :param axis: axis across which to perform softmax
    :return: x with values across axis softmaxed
    """
    x_max = numpy.max(x, axis=axis, keepdims=True)
    e_x = numpy.exp(x - x_max)
    e_x_sum = numpy.sum(e_x, axis=axis, keepdims=True)
    softmax_x = e_x / e_x_sum
    return softmax_x

# TODO temperature
def choose_point_to_optimize(loss: MSELoss, temperature: int = 150):
    point_losses = loss.calc_point_losses()
    p = numpy_softmax(numpy.array(point_losses) / temperature)

    choice_loss = numpy.random.choice(point_losses, p=p)
    point_index = point_losses.index(choice_loss)

    return points[point_index]

def validate_points(points):
    num_points = len(points)

    if num_points <= 0:
        raise ValueError("No points to optimize")

    num_dims = len(points[0].position)

    for point in points:
        if len(point.position) != num_dims:
            raise ValueError(
                "All point positions must have the name number of dimensions"
            )

        if (
            point.target_distances is not None
            and len(point.target_distances) != (num_points)
        ):
            raise ValueError(
                "point target_distances must have length equal to the number "
                "of points"
            )

class Callback():
    def __init__(self, points: List[Point], ax, animate: bool = False, verbose: bool = False):
        self._points = points
        self._animate = animate
        self._verbose = verbose
        self._ax = ax
        self._texts = []

        if self._animate:
            self.init_animation(ax)

    def init_animation(self, ax):
        self.animation_frames = []
        self._ax.set_xlim(-1000, 1000)
        self._ax.set_ylim(-1000, 1000)

        self.scatter_plot = ax.scatter([], [], s=100)

        for point in self._points:
            self._texts.append(self._ax.annotate(point.name, point.position))

    def animate(self, _frame_i: int):
        offsets = numpy.array([point.position for point in self._points])
        self.scatter_plot.set_offsets(offsets)

        for text, offset in zip(self._texts, offsets):
            text.set_position(offset)

        return (self._ax, self.scatter_plot)

    def __call__(
        self,
        points: List[Point],
        point: Point,
        point_loss: float,
        total_loss: float
    ):
        self._points = points

        if self._verbose:
            print(
                f"point: {point} | loss: {point_loss:0.2f} | "
                f"total_loss: {total_loss:0.2f}"
            )

def _negate_values(values, max_value=200):
    return [200 - value for value in values]

def optimize_points(
    points: List[Point],
    learning_rate: float = 0.5,
    max_steps: int = 1000,
    minimum_loss: float = 0.0,
    temperature: float = 150.0,
    callback: Optional[Callback] = None,
):
    loss = MSELoss(points)
    optimizer = SimpleOptimizer(learning_rate=learning_rate)

    total_loss = loss.calc_total_loss()
    while total_loss > minimum_loss and optimizer.total_steps <= max_steps:
        point = choose_point_to_optimize(loss, temperature)
        point_loss = loss.calc_loss(point)

        gradient = loss.calc_gradient(point)
        optimizer.step(point, gradient)

        point_loss = loss.calc_loss(point)
        total_loss = loss.calc_total_loss()

        if callback:
            callback(points, point, point_loss, total_loss)

    return points

def plot_points(points):
    xs = [point.position[0] for point in points]
    ys = [point.position[1] for point in points]
    plt.scatter(xs, ys, s=100)
    for point in points:
        plt.text(*point.position, point.name)

    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    points = [  # TODO: ingest graph format
        Point([None] * 6 + _negate_values([136, 74, 30, 156, 72, 109, 42, 57]), name="Jumbo Kingdom"),
        Point([None] * 6 + _negate_values([75, 88, 22, 70, 106, 118, 42, 62]), name="World's Fair"),
        Point([None] * 6 + _negate_values([67, 103, 30, 83, 109, 78, 48, 43]), name="Jumbo Studios"),
        Point([None] * 6 + _negate_values([48, 44, 35, 70, 42, 52, 25, 18]), name="Animal Planet Zoo"),
        Point([None] * 6 + _negate_values([32, 44, 47, 43, 48, 23, 19, 16]), name="Trunk Water Park"),
        Point([None] * 6 + _negate_values([27, 17, 3, 17, 15, 18, 56, 32]), name="Jumbo Golf Course"),

        Point(_negate_values([136, 75, 67, 48, 32, 27]) + [None] * 8, name="Tusk Hotel"),
        Point(_negate_values([74, 88, 103, 44, 44, 17]) + [None] * 8, name="Mammoth Motel"),
        Point(_negate_values([30, 22, 30, 35, 47, 3]) + [None] * 8, name="Elephant Lodge"),
        Point(_negate_values([156, 70, 83, 70, 43, 17]) + [None] * 8, name="Trunk Inn"),
        Point(_negate_values([72, 106, 109, 42, 48, 15]) + [None] * 8, name="Loxodon Lodge"),
        Point(_negate_values([109, 118, 78, 52, 23, 18]) + [None] * 8, name="Pachyderm Suites"),
        Point(_negate_values([42, 42, 48, 25, 19, 56]) + [None] * 8, name="Mouse Resort"),
        Point(_negate_values([57, 62, 43, 18, 16, 32]) + [None] * 8, name="Oliphant Camp"),
    ]

    validate_points(points)

    fig, ax = plt.subplots()
    callback = Callback(points, ax, animate=args.animate, verbose=args.verbose)

    ani = animation.FuncAnimation(fig, callback.animate, interval=20, blit=True, save_count=50)

    thread = threading.Thread(target=optimize_points, args=(
        points,
        args.learning_rate,
        args.max_steps,
        args.minimum_loss,
        150,
        callback,
    ))

    thread.start()

    plt.show()

    """
    optimize_points(
        points,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        minimum_loss=args.minimum_loss,
        callback=callback,
    )
    """

    thread.join()

    #plot_points(points)
