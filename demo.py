import numpy

import matplotlib.pyplot as plt

class Point():
    def __init__(self, target_distances=[], position=None, name=None):
        self.target_distances = numpy.array(target_distances)
        self.name = name

        if position:
            self.position = numpy.array(position, dtype=numpy.float32)
        else:
            self.position = get_random_position()

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
        if weighted:
            return numpy.average(
                self.calc_point_losses(),
                weights=[
                    numpy.count_nonzero(point.target_distances is not None)
                    for point in self._points
                ])
        else:
            return numpy.mean(self.calc_point_losses())

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

    def calc_gradient(self, point):
        """
        E = (D - t) ^ 2
        dE/dx = 2(D - t) * (dD/dx)
        dD/dx = (1/d)(x2 - x1)

        dE/dx = 2 * ((D - t) / D) * (x2 - x1)
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

def choose_point_to_optimize(loss):
    max_loss_index = numpy.argmax(loss.calc_point_losses())
    return points[max_loss_index]

def validate_points(points):
    num_points = len(points)

    if num_points <= 0:
        raise ValueError("No points to optimize")

    for point in points:
        pass
        # TODO: Check position dimensions match

        #if len(point.target_distances) != (num_points):
        #    raise ValueError("TODO")

def get_random_position():
    return numpy.random.random_sample((2, )) * 500

def _negate_values(values, max_value=200):
    return [200 - value for value in values]

def optimize_points(points, learning_rate=0.5, max_steps=1000, minimum_loss=0.0):
    loss = MSELoss(points)
    optimizer = SimpleOptimizer(learning_rate=learning_rate)

    total_loss = loss.calc_total_loss()
    while total_loss > minimum_loss and optimizer.total_steps <= max_steps:
        point = choose_point_to_optimize(loss)
        point_loss = loss.calc_loss(point)
        print(f"point: {point} | loss: {point_loss:02f} | total_loss: {total_loss:0.2f}")

        gradient = loss.calc_gradient(point)
        optimizer.step(point, gradient)

        point_loss = loss.calc_loss(point)
        total_loss = loss.calc_total_loss()

        print(f"point: {point} | loss: {point_loss:0.2f} | total_loss: {total_loss:0.2f}")

    return points

def plot_points(points):
    xs = [point.position[0] for point in points]
    ys = [point.position[1] for point in points]
    plt.scatter(xs, ys, s=100)
    for point in points:
        plt.text(*point.position, point.name)

    plt.show()

if __name__ == "__main__":
    minimum_loss = 0.0
    max_steps = 1000
    learning_rate = 0.5
    points = [
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

    optimize_points(
        points,
        learning_rate=learning_rate,
        max_steps=max_steps,
        minimum_loss=minimum_loss
    )

    plot_points(points)
