"""
Rectangle class.
It will be particularly useful for the AITT project for describing bounding boxes.
"""

import numpy as np

from src.geometry.contour import Contour


class Rectangle(Contour):
    """Rectangle class to manipulate rectangle object"""

    def __init__(self, points: np.ndarray) -> None:
        assert len(points) == 4
        super().__init__(points)

    @classmethod
    def from_center(
        cls, center: np.ndarray, width: float, height: float, angle: float = 0
    ):
        """Create a Rectangle object using the center point, width, height and angle.
        The angle is defined as the

        Args:
            center (np.ndarray): _description_
            width (float): _description_
            height (float): _description_
            angle (float, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        # compute the halves lengths
        half_width = width / 2
        half_height = height / 2

        # get center coordinates
        center_x, center_y = center[0], center[1]

        # get the rectangle coordinates
        top_left_corner = np.array([center_x - half_width, center_y + half_height])
        top_right_corner = np.array([center_x + half_width, center_y + half_height])
        bottom_left_corner = np.array([center_x - half_width, center_y - half_height])
        bottom_right_corner = np.array([center_x + half_width, center_y - half_height])
        points = np.array(
            [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]
        )

        rect = Rectangle(points)
        rect = rect.rotate(angle=angle, pivot=center)

        return rect
