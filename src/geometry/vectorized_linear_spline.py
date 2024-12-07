"""
Vectorized Curve class useful to describe any kind of vectorized curves
"""

import numpy as np

from src.geometry import LinearSpline, Vector


class VectorizedLinearSpline(LinearSpline):
    """VectorizedLinearSpline class:

    - it IS a linear spline
    - it HAS a vector since a vector IS a segment and however the curve CANNOT be
        a segment. The vector is thus an attribute in this class.
        It does inherit from Vector class.
    """

    def __init__(self, points, is_cast_int=False):
        super().__init__(points, is_cast_int)
        self.vector_extremities = Vector(points=np.array([points[0], points[1]]))

    @property
    def is_two_points_vector(self) -> bool:
        """Whether the VectorizedLinearSpline is just a two points vector or not

        Returns:
            bool: True or false
        """
        return np.array_equal(self.asarray, self.vector_extremities.asarray)

    def cardinal_direction(self, full: bool = False, level: int = 2) -> str:
        """Cardinal direction

        Args:
            full (bool, optional): True returns full text (South), False returns
                abbreviated text (S). Defaults to False.
            level (int, optional): Level of detail (3 = N/NNE/NE/ENE/E...
                2 = N/NE/E/SE... 1 = N/E/S/W). Defaults to 2.

        Returns:
            str: _description_
        """
        return self.vector_extremities.cardinal_direction(full=full, level=level)
