"""
Rectangle class.
It will be particularly useful for the AITT project for describing bounding boxes.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pymupdf

from src.geometry.shape.polygon import Polygon


class Rectangle(Polygon):
    """Rectangle class to manipulate rectangle object"""

    def __init__(self, points: np.ndarray | list, is_cast_int: bool = False) -> None:
        assert len(points) == 4
        super().__init__(points=points, is_cast_int=is_cast_int)

    @classmethod
    def unit(cls) -> Rectangle:
        """Create a unit Rectangle object

        Returns:
            Rectangle: new Rectangle object
        """
        return cls(points=[[0, 0], [0, 1], [1, 1], [1, 0]])

    @classmethod
    def from_center(
        cls, center: np.ndarray, width: float, height: float, angle: float = 0
    ) -> Rectangle:
        """Create a Rectangle object using the center point, width, height and angle.

        Args:
            center (np.ndarray): center point of the rectangle
            width (float): width of the rectangle
            height (float): height of the rectangle
            angle (float, optional): rotation angle for the rectangle. Defaults to 0.

        Returns:
            Rectangle: Rectangle object
        """
        # compute the halves lengths
        half_width = width / 2
        half_height = height / 2

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

    @classmethod
    def from_topleft(
        cls, topleft: np.ndarray, width: float, height: float, angle: float = 0
    ) -> Rectangle:
        """Create a Rectangle object using the top left point, width, height and angle.

        Args:
            topleft (np.ndarray): top left point of the rectangle
            width (float): width of the rectangle
            height (float): height of the rectangle
            angle (float, optional): rotation angle for the rectangle. Defaults to 0.

        Returns:
            Rectangle: Rectangle object
        """
        center = topleft + np.array([width, height]) / 2
        return cls.from_center(center=center, width=width, height=height)

    @classmethod
    def from_topleft_bottomright(
        cls, topleft: np.ndarray, bottomright: np.ndarray
    ) -> Rectangle:
        """_summary_

        Args:
            topleft (np.ndarray): _description_
            bottomright (np.ndarray): _description_

        Returns:
            Rectangle: _description_
        """
        width = bottomright[0] - topleft[0]
        height = bottomright[1] - topleft[1]
        return cls.from_topleft(topleft=topleft, width=width, height=height)

    @property
    def as_pymupdf_rect(self) -> pymupdf.Rect:
        """Get the pymupdf representation of the given Rectangle.
        Beware a pymupdf can only be straight or axis-aligned.

        See: https://pymupdf.readthedocs.io/en/latest/rect.html

        Returns:
            pymupdf.Rect: pymupdf axis-aligned Rect object
        """
        return pymupdf.Rect(x0=self.xmin, y0=self.ymin, x1=self.xmax, y1=self.ymax)

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
        shared_points = self.shared_approx_points(rect, margin_dist_error)
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
