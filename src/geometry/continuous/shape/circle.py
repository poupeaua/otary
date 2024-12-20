"""
Circle Geometric Object
"""

import math

import numpy as np

from src.geometry import Ellipse


class Circle(Ellipse):
    """Circle geometrical object"""

    def __init__(self, center: np.ndarray, radius: float):
        super().__init__(foci1=center, foci2=center, semi_major_axis=radius)
        self.center = center
        self.radius = radius

    @property
    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    @property
    def centroid(self) -> float:
        return self.center

    def curvature(self) -> float:
        """Curvature of circle is a constant and does not depend on a position of
        a point

        Returns:
            float: curvature value
        """
        return 1 / self.radius
