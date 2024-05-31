"""
Rectangle class.
It will be particularly useful for the AITT project for describing bounding boxes.
"""

from __future__ import annotations

import numpy as np

from src.geometry.contour import Contour


class Rectangle(Contour):
    """Rectangle class to manipulate rectangle object"""

    def __init__(self, points: np.ndarray) -> None:
        assert len(points) == 4
        super().__init__(points)

    @classmethod
    def from_center(
        cls, center: np.ndarray, dim: tuple[float], angle: float = 0
    ) -> Rectangle:
        """Create a Rectangle object using the center point, width, height and angle.
        The angle is defined as the

        Args:
            center (np.ndarray): center point of the rectangle
            dim (tuple[float]): dimension of the rectangle (width, height)
            angle (float, optional): _description_. Defaults to 0.

        Returns:
            Rectangle: Rectangle object
        """
        # compute the halves lengths
        half_width = dim[0] / 2
        half_height = dim[1] / 2

        # get center coordinates
        center_x, center_y = center[0], center[1]

        # get the rectangle coordinates
        points = np.array(
            [
                [center_x - half_width, center_y + half_height],
                [center_x + half_width, center_y + half_height],
                [center_x + half_width, center_y - half_height],
                [center_x - half_width, center_y - half_height],
            ]
        )

        rect = Rectangle(points)
        rect = rect.rotate(angle=angle, pivot=center)

        return rect
