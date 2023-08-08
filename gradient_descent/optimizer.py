import numpy

from models import Point

class Optimizer:
    pass

class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._prev_change = 0  # implicit scalar, not vector
        self.total_steps = 0

    def step(self, point: Point, gradient: numpy.ndarray):
        change = gradient * self._learning_rate + self._momentum * self._prev_change
        point.position -= change

        self._prev_change = change
        self.total_steps += 1
