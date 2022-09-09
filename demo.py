import numpy

"""
E = (D - t) ^ 2
dE/dx = 2(D - t) * (dD/dx)
dD/dx = (1/d)(x2 - x1)

dE/dx = ((D - t) / D) * (x2 - x1)
"""

num_dimensions = 1
learning_rate = 0.01

unoptimizable_point_indexes = [1]

class Point():
    def __init__(self, position, target_distances, can_optimize=True):
        self.position = numpy.array(position, dtype=numpy.float32)
        self.target_distances = numpy.array(target_distances)
        self.can_optimize = can_optimize

points = [
    Point([-1, 0], [None, 0]),
    Point([1, 0], [None, None], can_optimize=False),
]

def calc_error(point_position, target_distance):
    pass

for point_i, point in enumerate(points):
    other_point_positions = [point.position for point in points]
    other_point_positions = numpy.delete(other_point_positions, point_i, axis=0)
    print(other_point_positions)

    d = [numpy.linalg.norm(pos - point.position) for pos in other_point_positions]
    print(d)

    target_distances = point.target_distances.copy()
    target_distances = numpy.delete(target_distances, point_i, axis=0)
    print(target_distances)

    gradient = [
        (numpy.linalg.norm(point.position - target_point_position) - target_distances[target_point_i]) / numpy.linalg.norm(point.position - target_point_position) * (point.position - target_point_position)
        for target_point_i, target_point_position in enumerate(other_point_positions)
    ]
    print(gradient)
    gradient = numpy.average(gradient, axis=0)
    print(gradient)

    print(point.position)
    point.position -= gradient * learning_rate
    print(point.position)

    break

"""
    point_target_distances = target_distances[point_i]

    gradient = numpy.mean([])


    for dim in range(num_dimensions):

        for target in target_distances[]
            gradient =
"""
