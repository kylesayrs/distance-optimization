import numpy

from models import Point

class SimpleOptimizer():
    def __init__(self, learning_rate: float = 0.01):
        self._learning_rate = learning_rate
        self.total_steps = 0

    def step(self, point: Point, gradient: numpy.ndarray):
        point.position -= gradient * self._learning_rate
        self.total_steps += 1
