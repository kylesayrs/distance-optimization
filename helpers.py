from typing import List

import numpy

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


def choose_point_to_optimize(points: List[Point], loss: MSELoss, temperature: int = 150):
    point_losses = loss.calc_point_losses()
    p = numpy_softmax(numpy.array(point_losses) / temperature)

    choice_loss = numpy.random.choice(point_losses, p=p)
    point_index = point_losses.index(choice_loss)

    return points[point_index]


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
