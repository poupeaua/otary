"""
Axis Aligned Rectangle python file
"""

from __future__ import annotations

import pymupdf

from numpy.typing import NDArray

from otary.geometry.discrete.shape.rectangle import Rectangle


class AxisAlignedRectangle(Rectangle):
    """
    Axis Aligned Rectangle class that inherits from Rectangle.
    It defines a rectangle that is axis-aligned, meaning its sides are parallel
    to the X and Y axes.
    """

    def __init__(self, points: NDArray | list, is_cast_int: bool = False) -> None:
        super().__init__(points=points, is_cast_int=is_cast_int)

        if not self.is_axis_aligned:
            raise ValueError("The rectangle is not axis aligned")

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
        cls, center: NDArray, width: float, height: float, is_cast_int=False
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
            super().from_center(center, width, height, angle=0, is_cast_int=is_cast_int)
        )

    @property
    def as_pymupdf_rect(self) -> pymupdf.Rect:
        """Get the pymupdf representation of the given Rectangle.
        Beware a pymupdf can only be straight or axis-aligned.

        See: https://pymupdf.readthedocs.io/en/latest/rect.html

        Returns:
            pymupdf.Rect: pymupdf axis-aligned Rect object
        """
        return pymupdf.Rect(x0=self.xmin, y0=self.ymin, x1=self.xmax, y1=self.ymax)
