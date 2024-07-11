"""
Triangle class module
"""

import numpy as np

from src.geometry import Contour


class Triangle(Contour):
    """Triangle class"""

    def __init__(self, points: np.ndarray | list) -> None:
        assert len(points) == 3
        super().__init__(points)
