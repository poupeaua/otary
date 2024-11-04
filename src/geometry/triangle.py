"""
Triangle class module
"""

import numpy as np

from src.geometry import Contour


class Triangle(Contour):
    """Triangle class"""

    def __init__(self, points: np.ndarray | list, is_cast_int: bool = False) -> None:
        assert len(points) == 3
        super().__init__(points=points, is_cast_int=is_cast_int)
