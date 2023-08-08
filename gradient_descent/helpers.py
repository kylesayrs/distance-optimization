from typing import List, Optional, Tuple

import numpy
import matplotlib.pyplot as plt

from models import Point
from loss import MSELoss

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


def validate_points(points: List[Point]):
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

def plot_points(
    points: List[Point],
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
):
    figure, axes = plt.subplots(figsize=figsize)
    axes.set_aspect(1)

    xs = [point.position[0] for point in points]
    ys = [point.position[1] for point in points]
    axes.scatter(xs, ys, s=100)
    for point in points:
        axes.text(*point.position, point.name)

    if out_path:
        figure.savefig(out_path)
    else:
        plt.show()

def plot_loss(
    losses: List[float],
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
):
    figure, axes = plt.subplots(figsize=figsize)

    axes.plot(losses)

    if out_path:
        figure.savefig(out_path)
    else:
        plt.show()

def negate_values(values: List[float], max_value: int = 200):
    return [max_value - value for value in values]

def initialize_point_positions(
    points: List[Point],
    loc: float = 0.0,
    expected_range: float = 1.0,
    num_dims: int = 2
):
    [
        point.initialize_position(
            loc=loc,
            scale=expected_range,
            num_dims=num_dims,
        )
        for point in points
    ]
