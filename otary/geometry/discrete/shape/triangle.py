"""
Triangle class module
"""

import numpy as np

from otary.geometry import Polygon


class Triangle(Polygon):
    """Triangle class"""

    def __init__(self, points: np.ndarray | list, is_cast_int: bool = False) -> None:
        if len(points) != 3:
            raise ValueError("Cannot create a Triangle since it must have 3 points")
        super().__init__(points=points, is_cast_int=is_cast_int)
