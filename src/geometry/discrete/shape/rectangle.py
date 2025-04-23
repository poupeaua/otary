"""
Rectangle class.
It will be particularly useful for the AITT project for describing bounding boxes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pymupdf

from src.geometry import Polygon, Segment


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
        cls,
        center: np.ndarray,
        width: float,
        height: float,
        angle: float = 0,
        is_cast_int: bool = False,
    ) -> Rectangle:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
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

        rect = Rectangle(points=points, is_cast_int=is_cast_int)

        if angle != 0:
            rect = rect.rotate(angle=angle, pivot=center)
            if is_cast_int:
                rect.asarray = rect.asarray.astype(int)

        return rect

    @classmethod
    def from_topleft(
        cls,
        topleft: np.ndarray,
        width: float,
        height: float,
        angle: float = 0,
        is_cast_int: bool = False,
    ) -> Rectangle:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
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
        return cls.from_center(
            center=center,
            width=width,
            height=height,
            angle=angle,
            is_cast_int=is_cast_int,
        )

    @classmethod
    def from_topleft_bottomright(
        cls,
        topleft: np.ndarray,
        bottomright: np.ndarray,
        angle: float = 0,
        is_cast_int: bool = False,
    ) -> Rectangle:
        """Create a Rectangle object using the top left and bottom right points.

        Args:
            topleft (np.ndarray): top left point of the rectangle
            bottomright (np.ndarray): bottom right point of the rectangle

        Returns:
            Rectangle: new Rectangle object
        """
        width = bottomright[0] - topleft[0]
        height = bottomright[1] - topleft[1]
        return cls.from_topleft(
            topleft=topleft,
            width=width,
            height=height,
            angle=angle,
            is_cast_int=is_cast_int,
        )

    @property
    def is_axis_aligned(self) -> bool:
        """Check if the rectangle is axis-aligned

        Returns:
            bool: True if the rectangle is axis-aligned, False otherwise
        """
        if self.is_self_intersected:
            return False
        longside_cond = bool(
            (round(self.longside_slope_angle(degree=True)) + 90) % 90 == 0
        )
        shortside_cond = bool(
            (round(self.shortside_slope_angle(degree=True)) + 90) % 90 == 0
        )
        return longside_cond and shortside_cond

    @property
    def as_pymupdf_rect(self) -> pymupdf.Rect:
        """Get the pymupdf representation of the given Rectangle.
        Beware a pymupdf can only be straight or axis-aligned.

        See: https://pymupdf.readthedocs.io/en/latest/rect.html

        Returns:
            pymupdf.Rect: pymupdf axis-aligned Rect object
        """
        return pymupdf.Rect(x0=self.xmin, y0=self.ymin, x1=self.xmax, y1=self.ymax)

    @property
    def longside_length(self) -> float:
        """Compute the biggest side of the rectangle

        Returns:
            float: the biggest side length
        """
        seg1 = Segment(points=[self.points[0], self.points[1]])
        seg2 = Segment(points=[self.points[1], self.points[2]])
        return seg1.length if seg1.length > seg2.length else seg2.length

    @property
    def shortside_length(self) -> float:
        """Compute the smallest side of the rectangle

        Returns:
            float: the smallest side length
        """
        seg1 = Segment(points=[self.points[0], self.points[1]])
        seg2 = Segment(points=[self.points[1], self.points[2]])
        return seg2.length if seg1.length > seg2.length else seg1.length

    def longside_slope_angle(self, degree: bool = False, is_cv2: bool = False) -> float:
        """Compute the biggest slope of the rectangle

        Returns:
            float: the biggest slope
        """
        seg1 = Segment(points=[self.points[0], self.points[1]])
        seg2 = Segment(points=[self.points[1], self.points[2]])
        seg_bigside = seg1 if seg1.length > seg2.length else seg2
        return seg_bigside.slope_angle(degree=degree, is_cv2=is_cv2)

    def shortside_slope_angle(
        self, degree: bool = False, is_cv2: bool = False
    ) -> float:
        """Compute the smallest slope of the rectangle

        Returns:
            float: the smallest slope
        """
        seg1 = Segment(points=[self.points[0], self.points[1]])
        seg2 = Segment(points=[self.points[1], self.points[2]])
        seg_smallside = seg2 if seg1.length > seg2.length else seg1
        return seg_smallside.slope_angle(degree=degree, is_cv2=is_cv2)

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
        shared_points = self.shared_approx_vertices(rect, margin_dist_error)
        n_shared_points = len(shared_points)

        if n_shared_points in (0, 1):
            return None
        if n_shared_points == 2:
            new_rect_points = np.concatenate(
                (
                    self.vertices_far_from(shared_points, margin_dist_error),
                    rect.vertices_far_from(shared_points, margin_dist_error),
                ),
                axis=0,
            )
            return Rectangle(points=new_rect_points)
        # if 3 or more points in common it is the same rectangle
        return self
