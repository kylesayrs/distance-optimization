from typing import List, Optional

import numpy

class Point():
    def __init__(
        self,
        target_distances: List[float] = [],
        position: Optional[List[float]] = None,
        name: str = "",
    ):
        self.target_distances = numpy.array(target_distances)
        self.name = name

        if position:
            self.position = numpy.array(position, dtype=numpy.float32)
        else:
            self.position = None

    def initialize_position(self, loc=0.0, scale=1, num_dims=2):
        position = numpy.random.normal(loc=loc, scale=scale, size=(num_dims, ))
        self.position = position

    def __repr__(self):
        return str(self)

    def __str__(self):
        positions_string = ", ".join([f"{pos:0.2f}" for pos in self.position])
        if self.name:
            return f"Point(name=\"{self.name}\")"
        else:
            return f"Point(({positions_string}))"
