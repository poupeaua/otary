"""
Axis Aligned Rectangle python file
"""

from __future__ import annotations
from typing import Optional

import pymupdf
from numpy.typing import NDArray

from otary.geometry.discrete.shape.rectangle import Rectangle
from otary.geometry.utils.tools import rotate_2d_points


class AxisAlignedRectangle(Rectangle):
    """
    Axis Aligned Rectangle class that inherits from Rectangle.
    It defines a rectangle that is axis-aligned, meaning:
     - its sides are parallel to the X and Y axes
     - the first point is the top-left point (considering the y-axis pointing downwards)
     - the points are ordered clockwise
    """

    def __init__(self, points: NDArray | list, is_cast_int: bool = False) -> None:
        super().__init__(points=points, is_cast_int=is_cast_int)

        if not self.is_axis_aligned:
            raise ValueError(
                "Trying to create an AxisAlignedRectangle from a non-axis-aligned "
                "rectangle. Please check the input points."
            )

    @classmethod
    def from_rectangle(cls, rectangle: Rectangle) -> AxisAlignedRectangle:
        """Create an AxisAlignedRectangle from an ordinary Rectangle.
        Only works if the input Rectangle forms an AxisAlignedRectangle with its points.

        Args:
            rectangle (Rectangle): Rectangle object

        Returns:
            AxisAlignedRectangle: AxisAlignedRectangle object
        """
        return cls(points=rectangle.points)

    @classmethod
    def from_center(
        cls,
        center: NDArray,
        width: float,
        height: float,
        is_cast_int: bool = False,
    ) -> AxisAlignedRectangle:
        """Create an AxisAlignedRectangle from a center point

        Args:
            center (NDArray): center point of the rectangle
            width (float): width of the rectangle
            height (float): height of the rectangle
            is_cast_int (bool, optional): cast the points coordinates to int

        Returns:
            AxisAlignedRectangle: new AxisAlignedRectangle object
        """
        return cls.from_rectangle(
            super().from_center(center, width, height, is_cast_int=is_cast_int)
        )

    @property
    def height(self) -> float:
        """Get the height of the AxisAlignedRectangle

        Returns:
            float: height of the rectangle
        """
        return self.ymax - self.ymin

    @property
    def width(self) -> float:
        """Get the width of the AxisAlignedRectangle

        Returns:
            float: width of the rectangle
        """
        return self.xmax - self.xmin

    @property
    def area(self) -> float:
        """Get the area of the AxisAlignedRectangle

        Returns:
            float: area of the rectangle
        """
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        """Get the perimeter of the AxisAlignedRectangle

        Returns:
            float: perimeter of the rectangle
        """
        return 2 * (self.width + self.height)

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
    def rotated90(self) -> AxisAlignedRectangle:
        """Get the unique other related AxisAlignedRectangle that is the same one
        but rotated 90 degrees around its center.

        Returns:
            AxisAlignedRectangle: new AxisAlignedRectangle object
        """
        return AxisAlignedRectangle.from_center(
            center=self.centroid, width=self.height, height=self.width
        )

    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        pivot: Optional[NDArray] = None,
    ):
        raise TypeError("AxisAlignedRectangle does not support arbitrary rotation.")

    def rotate_transform(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        pivot: Optional[NDArray] = None,
    ) -> Rectangle:
        """Rotate the AxisAlignedRectangle and transform it into a Rectangle.
        The result is a Rectangle object since the result is not
        guaranteed to be axis-aligned anymore.

        Args:
            angle (float): rotation angle
            is_degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.
            is_clockwise (bool, optional): whether the rotation is clockwise or
                counter-clockwise. Defaults to True.
            pivot (NDArray, optional): pivot point.
                Defaults to None which means that by default the centroid point of
                the shape is taken as the pivot point.

        Returns:
            Rectangle: object resulting from the rotation

        Examples:
            The rotate method does not modify the original AxisAlignedRectangle object:
            >>> rect = ot.AxisAlignedRectangle(points=[[0, 0], [4, 0], [4, 3], [0, 3]])
            >>> rot_rect = rect.rotate(angle=45, is_degree=True)
            >>> rot_rect.asarray
            array([[ 1.6464466 , -0.9748737 ],
                   [ 4.4748735 ,  1.8535534 ],
                   [ 2.3535533 ,  3.9748738 ],
                   [-0.47487372,  1.1464466 ]], dtype=float32)
            >>> rect
            array([[0, 0],
                   [4, 0],
                   [4, 3],
                   [0, 3]], dtype=int32)
        """
        # pylint: disable=R0801
        if pivot is None:
            pivot = self.centroid

        rot_points = rotate_2d_points(
            points=self.points,
            angle=angle,
            pivot=pivot,
            is_degree=is_degree,
            is_clockwise=is_clockwise,
        )
        return Rectangle(points=rot_points)
