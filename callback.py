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
        self.losses = []


    def __call__(
        self,
        steps: int,
        points: List[Point],
        point: Point,
        point_loss: float,
        total_loss: float,
        temperature: float,
    ):
        if self._verbose:
            print(
                f"steps: {steps} | total_loss: {total_loss:0.4f} | "
                f"temp: {temperature:0.1f} | point: {point}"
            )

        if self._animator:
            self._animator.points = points

        self.losses.append(total_loss)
