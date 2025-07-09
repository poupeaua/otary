"""
Vectorized Curve class useful to describe any kind of vectorized curves
"""

import numpy as np

from otary.geometry.discrete.linear.directed.entity import DirectedLinearEntity
from otary.geometry import LinearSpline, Vector


class VectorizedLinearSpline(LinearSpline, DirectedLinearEntity):
    """VectorizedLinearSpline class:

    - it IS a linear spline
    - it HAS a vector since a vector IS a segment and however the curve CANNOT be
        a segment. The vector is thus an attribute in this class.
        The vector does inherit from Vector class.
    """

    def __init__(self, points, is_cast_int=False):
        super().__init__(points, is_cast_int)
        self.vector_extremities = Vector(points=np.array([points[0], points[-1]]))

    @property
    def is_simple_vector(self) -> bool:
        """Whether the VectorizedLinearSpline is just a two points vector or not

        Returns:
            bool: True or false
        """
        return np.array_equal(self.asarray, self.vector_extremities.asarray)

    @property
    def cardinal_degree(self) -> float:
        """Returns the cardinal degree of the VectorizedLinearSpline in the cv2 space.
        It is calculated using the two extremities points that compose the object.

        We consider the top of the image to point toward the north as default and thus
        represent the cardinal degree value 0 mod 360.

        Returns:
            float: cardinal degree
        """
        return self.vector_extremities.cardinal_degree
