from typing import List, Optional

import argparse
import numpy
import threading

from models import Point
from loss import MSELoss
from optimizer import SGD
from animator import Animator
from callback import Callback
from helpers import (
    initialize_point_positions,
    validate_points,
    numpy_softmax,
    negate_values,
    plot_points,
    plot_loss
)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--minimum_loss", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=30000)
parser.add_argument("--learning_rate", type=float, default=0.03)
parser.add_argument("--momentum", type=float, default=0.99)
parser.add_argument("--initial_temperature", type=float, default=500.0)
parser.add_argument("--change_temperature", type=float, default=-0.007)
parser.add_argument("--expected_range", type=float, default=500)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--animate", type=bool, default=True)

def choose_point_to_optimize(points: List[Point], loss: MSELoss, temperature: int = 150):
    point_losses = loss.calc_point_losses()
    p = numpy_softmax(numpy.array(point_losses) / max(temperature, 1))

    choice_loss = numpy.random.choice(point_losses, p=p)
    point_index = point_losses.index(choice_loss)

    return points[point_index]

def optimize_points(
    points: List[Point],
    learning_rate: float = 0.5,
    momentum: float = 0.0,
    max_steps: int = 5000,
    minimum_loss: float = 0.0,
    initial_temperature: float = 150.0,
    change_temperature: float = -1,
    expected_range: float = 1,
    callback: Optional[Callback] = None,
    **kwargs,
):
    loss = MSELoss(points)
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

    temperature = initial_temperature
    total_loss = loss.calc_total_loss()
    while total_loss > minimum_loss and optimizer.total_steps <= max_steps:
        point = choose_point_to_optimize(points, loss, temperature)
        point_loss = loss.calc_loss(point)

        gradient = loss.calc_gradient(point)
        optimizer.step(point, gradient)

        point_loss = loss.calc_loss(point)
        total_loss = loss.calc_total_loss()

        temperature += change_temperature
        temperature = max(temperature, 1)

        if callback:
            callback(
                optimizer.total_steps,
                points,
                point,
                point_loss,
                total_loss,
                temperature
            )

    return points

if __name__ == "__main__":
    args = parser.parse_args()

    points = []

    initialize_point_positions(points)
    validate_points(points)

    animator = Animator(points, expected_range=args.expected_range)
    callback = Callback(animator=animator, verbose=args.verbose)

    optimize_kwargs = vars(args)
    optimize_kwargs.update({"callback": callback})
    optimize_thread = threading.Thread(
        target=optimize_points,
        args=(points, ),
        kwargs=optimize_kwargs
    )

    optimize_thread.start()
    if animator:
        animator.show_animation()
    optimize_thread.join()

    plot_points(points)
    plot_loss(callback.losses)
