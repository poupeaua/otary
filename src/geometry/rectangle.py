"""
Rectangle class.
It will be particularly useful for the AITT project for describing bounding boxes.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from src.geometry.contour import Contour


class Rectangle(Contour):
    """Rectangle class to manipulate rectangle object"""

    def __init__(self, points: np.ndarray | list) -> None:
        assert len(points) == 4
        super().__init__(points)

    @classmethod
    def from_center(
        cls, center: np.ndarray, dim: tuple[float, float], angle: float = 0
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

    def join(
        self, rect: Rectangle, margin_dist_error: float = 5
    ) -> Optional[Rectangle]:
        """Join two rectangles into a single one.
        If they share no point in common or only a single point returns None.
        If they share two points, returns a new Rectangle that is the concatenation
        of the two rectangles.
        If they share 3 or more points they represent the same rectangle, thus
        returns this object.

        Args:
            rect (Rectangle): the other Rectangle object
            margin_dist_error (float, optional): the threshold to consider whether the
                rectangle share a common point. Defaults to 5.

        Returns:
            Rectangle: the join new Rectangle object
        """
        shared_points = self.shared_close_points(rect, margin_dist_error)
        n_shared_points = len(shared_points)

        if n_shared_points in (0, 1):
            return None
        if n_shared_points == 2:
            new_rect_points = np.concatenate(
                (
                    self.points_far_from(shared_points, margin_dist_error),
                    rect.points_far_from(shared_points, margin_dist_error),
                ),
                axis=0,
            )
            return Rectangle(points=new_rect_points)
        # if 3 or more points in common it is the same rectangle
        return self
