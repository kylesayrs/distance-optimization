from typing import List

import numpy

from models import Point

class MSELoss():
    def __init__(self, points: List[Point]):
        self._points = points

    def calc_total_loss(self, weighted: bool = False):
        weights=[
            numpy.count_nonzero(point.target_distances is not None)
            for point in self._points
        ] if weighted else None

        return numpy.average(self.calc_point_losses(), weights=weights)

    def calc_point_losses(self):
        return [self.calc_loss(point) for point in self._points]

    def calc_loss(self, point: Point) -> float:
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
