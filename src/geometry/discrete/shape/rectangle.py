"""
Rectangle class.
It will be particularly useful for the AITT project for describing bounding boxes.
"""

from __future__ import annotations

from typing import Optional, Self

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
        angle: float = 0.0,
        is_cast_int: bool = False,
    ) -> Rectangle:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Create a Rectangle object using the center point, width, height and angle.

        Args:
            center (np.ndarray): center point of the rectangle
            width (float): width of the rectangle
            height (float): height of the rectangle
            angle (float, optional): radian rotation angle for the rectangle.
                Defaults to 0.

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
                rect.asarray = np.round(rect.asarray).astype(int)

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

        precision = 3
        longside_cond = bool(
            (round(self.longside_slope_angle(degree=True), precision) + 90) % 90 == 0
        )
        shortside_cond = bool(
            (round(self.shortside_slope_angle(degree=True), precision) + 90) % 90 == 0
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
        if not self.is_axis_aligned:
            raise RuntimeError(
                "The rectangle is not axis-aligned, thus it cannot be converted to a "
                "pymupdf Rect object."
            )
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

    def desintersect(self) -> Self:
        """Desintersect the rectangle if it is self-intersected.
        If the rectangle is not self-intersected, returns the same rectangle.

        Returns:
            Rectangle: the desintersected Rectangle object
        """
        if not self.is_self_intersected:
            return self

        # Sort points based on angle from centroid
        def angle_from_center(pt):
            return np.arctan2(pt[1] - self.centroid[1], pt[0] - self.centroid[0])

        sorted_vertices = sorted(self.asarray, key=angle_from_center)
        self.asarray = np.array(sorted_vertices)
        return self

    def join(
        self, rect: Rectangle, margin_dist_error: float = 1e-5
    ) -> Optional[Rectangle]:
        """Join two rectangles into a single one.
        If they share no point in common or only a single point returns None.
        If they share two points, returns a new Rectangle that is the concatenation
        of the two rectangles and that is not self-intersected.
        If they share 3 or more points they represent the same rectangle, thus
        returns this object.

        Args:
            rect (Rectangle): the other Rectangle object
            margin_dist_error (float, optional): the threshold to consider whether the
                rectangle share a common point. Defaults to 1e-5.

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
            return Rectangle(points=new_rect_points).desintersect()
        # if 3 or more points in common it is the same rectangle
        return self

    def _topright_vertice_from_topleft(self, topleft_index: int) -> np.ndarray:
        """Get the top-right vertice from the topleft vertice

        Args:
            topleft_index (int): index of the topleft vertice

        Returns:
            np.ndarray: topright vertice
        """
        if self.is_clockwise(is_cv2=True):
            return self.asarray[(topleft_index + 1) % len(self)]
        else:
            return self.asarray[topleft_index - 1]

    def _bottomleft_vertice_from_topleft(self, topleft_index: int) -> np.ndarray:
        """Get the bottom-left vertice from the topleft vertice

        Args:
            topleft_index (int): index of the topleft vertice

        Returns:
            np.ndarray: topright vertice
        """
        if self.is_clockwise(is_cv2=True):
            return self.asarray[topleft_index - 1]
        else:
            return self.asarray[(topleft_index + 1) % len(self)]

    def _bottomright_vertice_from_topleft(self, topleft_index: int) -> np.ndarray:
        """Get the bottom-right vertice from the topleft vertice

        Args:
            topleft_index (int): index of the topleft vertice

        Returns:
            np.ndarray: topright vertice
        """
        return self.asarray[(topleft_index + 2) % len(self)]

    def vertice_from_topleft(
        self, topleft_index: int, vertice: str = "topright"
    ) -> np.ndarray:
        """Get vertice from the topleft vertice. You can use this method to
        obtain the topright, bottomleft, bottomright vertice from the topleft vertice.

        Returns:
            np.ndarray: topright vertice
        """
        if vertice not in ("topright", "bottomleft", "bottomright"):
            raise ValueError(
                "Parameter vertice must be one of"
                "'topright', 'bottomleft', 'bottomright'"
                f"but got {vertice}"
            )
        return getattr(self, f"_{vertice}_vertice_from_topleft")(topleft_index)

    def width_from_topleft(self, topleft_index: int) -> float:
        return np.linalg.norm(
            self.asarray[topleft_index]
            - self.vertice_from_topleft(topleft_index, "topright")
        )

    def heigth_from_topleft(self, topleft_index: int) -> float:
        return np.linalg.norm(
            self.asarray[topleft_index]
            - self.vertice_from_topleft(topleft_index, "bottomleft")
        )
