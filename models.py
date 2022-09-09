from typing import List, Optional

import numpy

class Point():
    def __init__(
        self,
        target_distances: List[float] = [],
        position: Optional[List[float]] = None,
        name: str = "",
        num_dims: int = 2,
        init_scale: float = 500,
    ):
        self.target_distances = numpy.array(target_distances)
        self.name = name
        self._num_dims = num_dims
        self._init_scale = init_scale

        if position:
            self.position = numpy.array(position, dtype=numpy.float32)
        else:
            self.position = self.initialize_position()

    def initialize_position(self):
        position = numpy.random.normal(
            loc=0.0,
            scale=self._init_scale,
            size=(self._num_dims, )
        )
        self.position = position
        return position

    def __repr__(self):
        return str(self)

    def __str__(self):
        positions_string = ", ".join([f"{pos:0.2f}" for pos in self.position])
        if self.name:
            return f"Point(name=\"{self.name}\", ({positions_string}))"
        else:
            return f"Point(({positions_string}))"
