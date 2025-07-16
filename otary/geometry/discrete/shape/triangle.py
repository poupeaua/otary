"""
Triangle class module
"""

from __future__ import annotations

import numpy as np

from otary.geometry import Polygon


class Triangle(Polygon):
    """Triangle class"""

    def __init__(self, points: np.ndarray | list, is_cast_int: bool = False) -> None:
        if len(points) != 3:
            raise ValueError("Cannot create a Triangle since it must have 3 points")
        super().__init__(points=points, is_cast_int=is_cast_int)

    def __str__(self) -> str:
        return (  # pylint: disable=duplicate-code
            self.__class__.__name__
            + "(["
            + self.asarray[0].tolist().__str__()
            + ", "
            + self.asarray[1].tolist().__str__()
            + ", "
            + self.asarray[2].tolist().__str__()
            + "])"
        )

    def __repr__(self) -> str:
        return str(self)
