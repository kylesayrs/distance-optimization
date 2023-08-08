from typing import List, Tuple

import argparse
import numpy
import threading

from main import optimize_points
from models import Point
from animator import Animator
from callback import Callback
from helpers import (
    initialize_point_positions,
    validate_points,
    plot_points,
    plot_loss
)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("num_points", type=int)
parser.add_argument("--minimum_loss", type=float, default=0.01)
parser.add_argument("--max_steps", type=int, default=30000)
parser.add_argument("--learning_rate", type=float, default=0.07)
parser.add_argument("--momentum", type=float, default=0.99)
parser.add_argument("--initial_temperature", type=float, default=1000.0)
parser.add_argument("--change_temperature", type=float, default=-0.08)
parser.add_argument("--expected_range", type=float, default=500)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--silent', dest='verbose', action='store_false')
parser.add_argument('--animate', dest='animate', action='store_true')
parser.add_argument('--no_animate', dest='animate', action='store_false')
parser.set_defaults(verbose=True, animate=True)

def points_from_positions(point_positions: List[Tuple[float, float]]):
    points = []
    for point_i, point_position in enumerate(point_positions):
        target_distances = [
            numpy.linalg.norm(point_position - target_position)
            for target_position in point_positions
        ]
        target_distances[point_i] = None

        points.append(Point(target_distances, name=str(point_i)))

    return points

if __name__ == "__main__":
    args = parser.parse_args()

    point_true_positions = [
        numpy.array([
            numpy.random.randint(-1 * args.expected_range, args.expected_range),
            numpy.random.randint(-1 * args.expected_range, args.expected_range)
        ])
        for _ in range(args.num_points)
    ]
    points = points_from_positions(point_true_positions)

    initialize_point_positions(points)
    validate_points(points)

    animator = (
        Animator(points, expected_range=args.expected_range)
        if args.animate else None
    )
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
