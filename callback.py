from typing import List, Optional

from models import Point
from animator import Animator

class Callback():
    def __init__(
        self,
        animator: Optional[Animator] = None,
        verbose: bool = False,
    ):
        self._animator = animator
        self._verbose = verbose


    def __call__(
        self,
        points: List[Point],
        point: Point,
        point_loss: float,
        total_loss: float
    ):
        if self._verbose:
            print(
                f"point: {point} | loss: {point_loss:0.2f} | "
                f"total_loss: {total_loss:0.2f}"
            )

        if self._animator:
            self._animator.points = points
