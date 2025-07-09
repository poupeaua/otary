"""
Vectors class they are like segments, but with a given direction
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from otary.geometry.discrete.linear.directed.entity import DirectedLinearEntity
from otary.geometry import Segment


class Vector(Segment, DirectedLinearEntity):
    """Vector class to manipulate vector which can be seen as Segment with direction"""

    @classmethod
    def from_single_point(cls, point: NDArray) -> Vector:
        """Get vector that goes from [0, 0] to point

        Args:
            point (NDArray): point of shape 2

        Returns:
            Vector: new vector object
        """
        return cls(points=[[0, 0], point])

    @property
    def cardinal_degree(self) -> float:
        """Returns the cardinal degree of the vector in the cv2 space.
        We consider the top of the image to point toward the north as default and thus
        represent the cardinal degree value 0 mod 360.

        Returns:
            float: cardinal degree
        """
        angle = self.slope_angle(degree=True, is_y_axis_down=True)

        # if angle is negative
        is_neg_sign_angle = bool(np.sign(angle) - 1)
        if is_neg_sign_angle:
            angle = 90 + np.abs(angle)
        else:
            angle = 90 - angle

        # if vector points towards west
        if self.is_x_first_pt_gt_x_last_pt:
            angle += 180

        cardinal_degree = np.mod(360 + angle, 360)  # avoid negative value case
        return cardinal_degree

    @property
    def coordinates_shift(self) -> NDArray:
        """Return the vector as a single point (x1-x0, y1-y0)

        Returns:
            NDArray: coordinates shift
        """
        return self.origin[1]

    @property
    def normalized(self) -> NDArray:
        """Nornalized vector

        Returns:
            NDArray: normalized vector
        """
        return self.coordinates_shift / np.linalg.norm(self.coordinates_shift) + 1e-9

    def rescale_head(self, scale: float) -> Vector:
        """Rescale the head part of the vector without moving the first point.
        This method only updates the second point that composes the vector.

        Args:
            scale (float): scale factor

        Returns:
            Vector: scaled vector
        """
        self.asarray = (self.asarray - self.tail) * scale + self.tail
        return self
