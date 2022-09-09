import numpy

class Point():
    def __init__(self, position, target_distances):
        self.position = numpy.array(position, dtype=numpy.float32)
        self.target_distances = numpy.array(target_distances)

    def __repr__(self):
        return str(self)

    def __str__(self):
        positions_string = ", ".join([f"{pos:0.2f}" for pos in self.position])
        return f"Point ({positions_string})"

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

    def step(self, point, gradient):
        point.position -= gradient * self._learning_rate

def choose_point_to_optimize(loss):
    max_loss_index = numpy.argmax(loss.calc_point_losses())
    return points[max_loss_index]

def validate_points(points):
    num_points = len(points)

    if num_points <= 0:
        raise ValueError("No points to optimize")

    for point in points:
        # TODO: Check position dimensions match

        if len(point.target_distances) != (num_points):
            raise ValueError("TODO")

if __name__ == "__main__":
    minimum_loss = 0.000001
    points = [
        Point([-1], [None, None, 0]),
        Point([7], [None, None, 0]),
        Point([1], [None, None, None]),
    ]

    loss = MSELoss(points)
    optimizer = SimpleOptimizer(learning_rate=0.5)

    validate_points(points)

    total_loss = loss.calc_total_loss()
    while total_loss > minimum_loss:
        point = choose_point_to_optimize(loss)

        gradient = loss.calc_gradient(point)
        optimizer.step(point, gradient)

        total_loss = loss.calc_total_loss()

        print(f"points: {points} | loss: {total_loss:0.2f}")
