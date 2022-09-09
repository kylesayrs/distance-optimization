from typing import List, Optional

import argparse
import numpy
import threading

from models import Point
from loss import MSELoss
from optimizer import SGD
from animator import Animator
from callback import Callback
from helpers import validate_points, choose_point_to_optimize, plot_points, plot_loss

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--minimum_loss", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.03)
parser.add_argument("--momentum", type=float, default=0.99)
parser.add_argument("--initial_temperature", type=float, default=500.0)
parser.add_argument("--change_temperature", type=float, default=-0.007)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--animate", type=bool, default=True)

def _negate_values(values: List[float], max_value: int = 200):
    return [200 - value for value in values]

def optimize_points(
    points: List[Point],
    learning_rate: float = 0.5,
    momentum: float = 0.0,
    max_steps: int = 5000,
    minimum_loss: float = 0.0,
    initial_temperature: float = 150.0,
    change_temperature: float = -1,
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

    animator = Animator(points, 500) if args.animate else None
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
